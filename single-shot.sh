#!/usr/bin/env bash
#
# fork the server for a single shot client call
# use enviornment to adjust parameters
# 


PORT=${PORT:-9002}
INFILE=${INFILE:-in/example.h5}
OUTFILE=${OUTFILE:-/tmp/examle.h5}
CONFIG=${CONFIG:-invertcontrast}
OUTGROUP=${OUTGROUP:-dataset} # ismrmr's default

scriptdir="$(cd $(dirname "$0"); pwd)"

# Make output directory if needed
mkdir -p "$(dirname "$OUTFILE")"

python $scriptdir/main.py -p $PORT &
pid_server=$!
python $scriptdir/client.py "$INFILE" -p $PORT -o "$OUTFILE" -c $CONFIG -G "$OUTGROUP"

kill $pid_server
wait $pid_server

