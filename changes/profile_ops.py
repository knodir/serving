"""
Create individual profile of different TensorFlow operations.

Usage:
    python profile_ops.py -o MatMul -n <num-of-runs>
    Runs MatMul operation n times and reports the average runtime.

TODO:
"""
#!usr/bin/env python2.7

import argparse
import tensorflow as tf
import glog
import sys
import timeit


with tf.device('/cpu:0'):
    var1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='var1')
    var2 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='var2')
    res = tf.matmul(var1, var2)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def run_op(op_name, runs):
    """TBD.
    """
    timing = timeit.timeit(stmt='sess.run(res)',
            setup='from __main__ import sess, res',
            number=runs)
    glog.info('completed %d runs in %s sec', runs, timing)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, default="MatMul",
            help="Name of the operation")
    parser.add_argument("-n", type=int, default=1,
            help="Number of iterations to run the operation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    op_name = args.o
    num_of_runs = args.n
    glog.info('op_name: %s, num_of_runs: %d', op_name, num_of_runs)

    run_op(op_name, num_of_runs)
