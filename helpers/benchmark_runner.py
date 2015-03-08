"""Given experimental parameters, runs the required experiments and obtains the data
"""
import argparse
import ConfigParser
import datetime
import os

VALID_PARAMS = {"lambda", "minpart", "samplefrac", "oraclesize"}
VAL_PARAMS = {"lambda", "minpart", "samplefrac", "oraclesize", "stopcrit", "roundlimit", "gaplimit", "gapcheck",
              "timelimit", "debugmult"}
BOOL_PARAMS = {"sparse", "debug", "linesearch"}


def main():
    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument("identity_file", help="SSH private key to log into spark nodes")
    parser.add_argument("master_uri", help="URI of master node")
    parser.add_argument("expt_config", help="Experimental config file")
    args = parser.parse_args()

    config = ConfigParser.ConfigParser()
    config.read(args.expt_config)

    expt_name = config.get("general", "experiment_name")
    class_name = config.get("general", "class_name")
    num_repetitions = config.getint("general", "repetitions")

    # Pivot values
    pivot_param = config.get("pivot", "param")
    assert (pivot_param in VALID_PARAMS)
    pivot_values_raw = config.get("pivot", "values")
    pivot_values = map(lambda x: x.strip(), pivot_values_raw.split(","))

    # Paths
    lib_jar_path = config.get("paths", "lib_jar_path")
    examples_jar_path = config.get("paths", "examples_jar_path")
    spark_dir = config.get("paths", "spark_dir")
    spark_submit_path = os.path.join(spark_dir, "bin", "spark-submit")
    hdfs_input_path = config.get("paths", "hdfs_input_path")
    local_output_dir = config.get("paths", "local_output_dir")

    dtf = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    appname_format = "{dtf}-{expt_name}-R{rep_num}-{param}-{paramval}"
    spark_submit_cmd_format = ("{spark_submit} "
                               "--jars {lib_jar_path} "
                               "--class \"{class_name}\" "
                               "{spark_args} "
                               "{examples_jar_path} "
                               "{solver_options_args} "
                               "--kwargs k1=v1,{app_args}")

    for rep_num in range(1, num_repetitions + 1):
        for pivot_val in pivot_values:
            '''
            Construct command to execute on spark cluster
            '''
            appname = appname_format.format(dtf=dtf,
                                            expt_name=expt_name,
                                            rep_num=rep_num,
                                            param=pivot_param,
                                            paramval=pivot_val)

            # === Construct Spark arguments ===
            spark_args = ','.join(["--%s %s" % (k, v) for k, v in config.items("spark_args")])

            # === Construct Solver Options arguments ===
            valued_parameter_args = ' '.join(
                ["--%s %s" % (k, v) for k, v in config.items("parameters") if k in VAL_PARAMS])
            boolean_parameter_args = ' '.join(
                ["--%s" % k for k, v in config.items("parameters") if k in BOOL_PARAMS and bool(v)])
            valued_dissolve_args = ' '.join(
                ["--%s %s" % (k, v) for k, v in config.items("dissolve_args") if k in VAL_PARAMS])
            boolean_dissolve_args = ' '.join(
                ["--%s" % k for k, v in config.items("dissolve_args") if k in BOOL_PARAMS and bool(v)])

            # === Add the pivotal parameter ===
            assert (pivot_param not in config.options("parameters"))
            pivot_param_arg = "--%s %s" % (pivot_param, pivot_val)

            solver_options_args = ' '.join(
                [valued_parameter_args, boolean_parameter_args, valued_dissolve_args, boolean_dissolve_args,
                 pivot_param_arg])

            # == Construct App-specific arguments ===
            default_app_args = ("appname={appname},"
                                "input_path={input_path}").format(appname=appname,
                                                                  input_path=hdfs_input_path)
            extra_app_args = ','.join(["%s=%s" % (k, v) for k, v in config.items("app_args")])

            app_args = ','.join([default_app_args, extra_app_args])

            spark_submit_cmd = spark_submit_cmd_format.format(spark_submit=spark_submit_path,
                                                              lib_jar_path=lib_jar_path,
                                                              class_name=class_name,
                                                              examples_jar_path=examples_jar_path,
                                                              spark_args=spark_args,
                                                              solver_options_args=solver_options_args,
                                                              app_args=app_args)

            print spark_submit_cmd
            print

            '''
            Execute Command
            '''

            '''
            Obtain required files
            '''

            '''
            Perform clean-up
            '''


if __name__ == '__main__':
    main()