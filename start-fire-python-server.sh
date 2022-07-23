#!/bin/bash

# Bash script to start Python ISMRMRD server
#
# First argument is path to log file.  If no argument is provided,
# logging is done to stdout (and discarded)

# Set Python's default temp folder to one that's shared with the host so that
# it's less likely to accidentally fill up the chroot
export TMPDIR=/tmp/share

if [ $# -eq 1 ]; then
  LOG_FILE=${1}
  python3 /opt/code/python-ismrmrd-server/main.py -v -r -H=0.0.0.0 -p=9002 -l=${LOG_FILE} &
else
  python3 /opt/code/python-ismrmrd-server/main.py -v -r -H=0.0.0.0 -p=9002 &
fi

