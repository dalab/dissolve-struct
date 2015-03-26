"""Given experimental parameters, runs the required experiments and obtains the data.
To be executed on the Hadoop main node.
"""
import argparse
import ConfigParser
import datetime
import re

from benchmark_utils import *
from paths import *

VALID_PARAMS = {"lambda", "minpart", "samplefrac", "oraclesize", "num-executors"}
VAL_PARAMS = {"lambda", "minpart", "samplefrac", "oraclesize", "stopcrit", "roundlimit", "gaplimit", "gapcheck", "gapthresh",
              "timelimit", "debugmult"}
BOOL_PARAMS = {"sparse", "debug", "linesearch"}

HOME_DIR = os.getenv("HOME")
PROJ_DIR = os.path.join(HOME_DIR, "dissolve-struct")

DEFAULT_CORES = 4


def execute(command, cwd=PROJ_DIR):
    subprocess.check_call(command, cwd=cwd, shell=True)


def str_to_bool(s):
    if s in ['True', 'true']:
        return True
    elif s in ['False', 'false']:
        return False
    else:
        raise ValueError("Boolean value in config '%s' unrecognized")


def main():
    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument("expt_config", help="Experimental config file")
    parser.add_argument("--ds", help="Run with debugging separately. Forces execution of two spark jobs",
                        action='store_true')
    args = parser.parse_args()

    # Check if setup has been executed
    touchfile_path = os.path.join(HOME_DIR, 'onesmallstep')
    execute("if [ ! -f %s ]; then echo \"Run benchmark_setup and try again\"; exit 1; fi" % touchfile_path,
            cwd=HOME_DIR)

    dtf = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    appname_format = "{dtf}-{expt_name}-{param}-{paramval}"
    spark_submit_cmd_format = ("{spark_submit} "
                               "--jars {lib_jar_path} "
                               "--class \"{class_name}\" "
                               "{spark_args} "
                               "{examples_jar_path} "
                               "{solver_options_args} "
                               "--kwargs k1=v1,{app_args}")

    config = ConfigParser.ConfigParser()
    config.read(args.expt_config)

    expt_name = config.get("general", "experiment_name")
    class_name = config.get("general", "class_name")

    if args.ds:
        assert ('debug' in config.options('dissolve_args'))
        # Want debug set to true, in case of double-debugging
        assert (str_to_bool(config.get('dissolve_args', 'debug')))

    # Pivot values
    pivot_param = config.get("pivot", "param")
    assert (pivot_param in VALID_PARAMS)
    pivot_values_raw = config.get("pivot", "values")
    pivot_values = map(lambda x: x.strip(), pivot_values_raw.split(","))

    # Paths
    examples_jar_path = EXAMPLES_JAR_PATH
    spark_submit_path = "spark-submit"
    input_path = config.get("paths", "input_path")

    output_dir = EXPT_OUTPUT_DIR
    local_output_expt_dir = os.path.join(output_dir, "%s_%s" % (expt_name, dtf))
    if not os.path.exists(local_output_expt_dir):
        os.makedirs(local_output_expt_dir)

    dissolve_lib_jar_path = LIB_JAR_PATH
    scopt_jar_path = SCOPT_JAR_PATH
    lib_jar_path = ','.join([dissolve_lib_jar_path, scopt_jar_path])

    '''
    Execute experiment
    '''
    for pivot_val in pivot_values:
        print "=== %s = %s ===" % (pivot_param, pivot_val)
        '''
        Construct command to execute on spark cluster
        '''
        appname = appname_format.format(dtf=dtf,
                                        expt_name=expt_name,
                                        param=pivot_param,
                                        paramval=pivot_val)

        # === Construct Solver Options arguments ===
        valued_parameter_args = ' '.join(
            ["--%s %s" % (k, v) for k, v in config.items("parameters") if k in VAL_PARAMS and k not in ['minpart']])
        # Treat 'minpart' as a special case. If minpart = 'auto', set minpart = num_cores * num_executors
        if 'minpart' in config.options('parameters'):
            if config.get('parameters', 'minpart') == 'auto':
                if pivot_param == 'num-executors':
                    num_executors = int(pivot_val)
                else:
                    num_executors = config.getint('spark_args', 'num-executors')
                minpart = DEFAULT_CORES * num_executors
            else:
                minpart = config.getint('parameters', 'minpart')
            minpart_arg = '--minpart %d' % minpart
            valued_parameter_args = ' '.join([valued_parameter_args, minpart_arg])
        boolean_parameter_args = ' '.join(
            ["--%s" % k for k, v in config.items("parameters") if k in BOOL_PARAMS and str_to_bool(v)])
        valued_dissolve_args = ' '.join(
            ["--%s %s" % (k, v) for k, v in config.items("dissolve_args") if k in VAL_PARAMS])
        boolean_dissolve_args = ' '.join(
            ["--%s" % k for k, v in config.items("dissolve_args") if k in BOOL_PARAMS and str_to_bool(v)])

        solver_options_args = ' '.join(
            [valued_parameter_args, boolean_parameter_args, valued_dissolve_args, boolean_dissolve_args])

        # === Construct Spark arguments ===
        spark_args = ' '.join(["--%s %s" % (k, v) for k, v in config.items("spark_args")])

        # === Add the pivotal parameter ===
        assert (pivot_param not in config.options("parameters"))
        assert (pivot_param not in config.options("spark_args"))
        pivot_param_arg = "--%s %s" % (pivot_param, pivot_val)

        # Is this pivotal parameters a spark argument or a dissolve argument?
        if pivot_param in ['num-executors', ]:
            spark_args = ' '.join([spark_args, pivot_param_arg])
        else:
            solver_options_args = ' '.join([solver_options_args, pivot_param_arg])

        # == Construct App-specific arguments ===
        debug_filename = "%s.csv" % appname
        debug_file_path = os.path.join('', debug_filename)
        default_app_args = ("appname={appname},"
                            "input_path={input_path},"
                            "debug_file={debug_file_path}").format(appname=appname,
                                                                   input_path=input_path,
                                                                   debug_file_path=debug_file_path)
        extra_app_args = ','.join(["%s=%s" % (k, v) for k, v in config.items("app_args")])

        app_args = ','.join([default_app_args, extra_app_args])

        spark_submit_cmd = spark_submit_cmd_format.format(spark_submit=spark_submit_path,
                                                          lib_jar_path=lib_jar_path,
                                                          class_name=class_name,
                                                          examples_jar_path=examples_jar_path,
                                                          spark_args=spark_args,
                                                          solver_options_args=solver_options_args,
                                                          app_args=app_args)

        '''
        Execute Command
        '''
        print "Executing:\n%s" % spark_submit_cmd
        execute(spark_submit_cmd)

        '''
        If enabled, execute command again, but without the debug flag
        '''
        if args.ds:
            no_debug_appname = appname + '.no_debug'
            debug_filename = "%s.csv" % no_debug_appname
            debug_file_path = os.path.join('', debug_filename)
            default_app_args = ("appname={appname},"
                                "input_path={input_path},"
                                "debug_file={debug_file_path}").format(appname=no_debug_appname,
                                                                       input_path=input_path,
                                                                       debug_file_path=debug_file_path)

            extra_app_args = ','.join(["%s=%s" % (k, v) for k, v in config.items("app_args")])

            app_args = ','.join([default_app_args, extra_app_args])

            # Get rid of the debugging flag
            solver_options_args = re.sub(' --debug$', ' ', solver_options_args)
            solver_options_args = re.sub(' --debug ', ' ', solver_options_args)

            no_debug_spark_submit_cmd = spark_submit_cmd_format.format(spark_submit=spark_submit_path,
                                                                      lib_jar_path=lib_jar_path,
                                                                      class_name=class_name,
                                                                      examples_jar_path=examples_jar_path,
                                                                      spark_args=spark_args,
                                                                      solver_options_args=solver_options_args,
                                                                      app_args=app_args)
            print "Executing WITHOUT debugging:\n%s" % no_debug_spark_submit_cmd
            execute(no_debug_spark_submit_cmd)


if __name__ == '__main__':
    main()