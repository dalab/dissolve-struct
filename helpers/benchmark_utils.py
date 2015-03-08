"""Common utils by benchmark scripts
"""
import subprocess


def ssh(host, username, identity_file, command):
    subprocess.check_call(
        "ssh -t -o StrictHostKeyChecking=no -i %s %s@%s '%s'" %
        (identity_file, username, host, command), shell=True)


# Copy a file to a given host through scp, throwing an exception if scp fails
def scp_to(host, identity_file, username, local_file, remote_file):
    subprocess.check_call(
        "scp -q -o StrictHostKeyChecking=no -i %s '%s' '%s@%s:%s'" %
        (identity_file, local_file, username, host, remote_file), shell=True)


# Copy a file to a given host through scp, throwing an exception if scp fails
def scp_from(host, identity_file, username, remote_file, local_file):
    subprocess.check_call(
        "scp -q -o StrictHostKeyChecking=no -i %s '%s@%s:%s' '%s'" %
        (identity_file, username, host, remote_file, local_file), shell=True)