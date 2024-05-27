#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Provide a tracefile as an argument"
    exit 1
fi

trace_file=$(realpath $1)
kdir="../../kutrace_min/"

cat $trace_file | $kdir/rawtoevent | sort -n | $kdir/eventtospan3 | LC_ALL=C sort > $trace_file.json 2>&1
cat $trace_file.json | $kdir/spantotrim test > $trace_file.trimmed.json
pushd $kdir
cat $trace_file.trimmed.json | ./makeself show_cpu.html > $trace_file.trimmed.html 2>&1
popd
