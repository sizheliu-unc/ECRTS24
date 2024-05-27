#!/bin/bash
set -e
trace=""
name=""

process () {
    trace_file=$(realpath $trace)
    trace_dir=$(realpath ../trace)
    kdir="../../kutrace_min"
    cat $trace_file | $kdir/rawtoevent | sort -n | $kdir/eventtospan3 | LC_ALL=C sort | $kdir/spantotrim test > $trace_file.trimmed.json
    pushd $kdir
    cat $trace_file.trimmed.json | ./makeself show_cpu.html > $trace_dir/$name.html 2>&1
    popd
    rm -f $trace_file
    rm -f $trace_file.trimmed.json
}

run_mutex () {
    trace="mutex_test_result.trace"
    sudo ../bin/mutex_test ni 1
    name="mutex_test_1_thread_ni"
    process
    sudo ../bin/mutex_test ni 2
    name="mutex_test_2_thread_ni"
    process
    sudo ../bin/mutex_test pi 1
    name="mutex_test_1_thread_pi"
    process
    sudo ../bin/mutex_test pi 2
    name="mutex_test_2_thread_pi"
    process
}

run_rwlock () {
    trace="rwlock_test_result.trace"
    sudo ../bin/rwlock_test 1
    name="rwlock_test_1_thread"
    process
    sudo ../bin/rwlock_test 4
    name="rwlock_test_4_thread"
    process
    sudo ../bin/rwlock_test_patched 1
    name="rwlock_test_1_thread_patched"
    process
    sudo ../bin/rwlock_test_patched 4
    name="rwlock_test_4_thread_patched"
    process
}

run_mutex
run_rwlock

