from __future__ import print_function

import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants

from google.protobuf import text_format

import argparse
import random
import os
import sys

DEFAULT_MODEL_PATH = "/tmp/inception-export/1/saved_model.pb"
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "random_placement")

CPU0 = "/device:CPU:0"

def print_nodes(sm):
    """Count the nodes in a saved_model"""
    count = 0
    for n in sm.meta_graphs[0].graph_def.node:
        count += 1
    print(count)

def print_ops(sm):
    """Print a list of ops."""
    ops = {}
    for node in sm.meta_graphs[0].graph_def.node:
        if node.op not in ops:
            ops[node.op] = 1
        else:
            ops[node.op] += 1
    for op in ops:
        print("{}:{}".format(op, ops[op]))
    print("total: {} distinct ops".format(len(ops.keys())))

def rand_assign(sm, output_dir):
    """Randomly assign nodes to CPU0 and save the modified model.

    Will create 5 variants with 10%, 20%, 30%, 40%, 50% of nodes assigned to
    CPU arbitrarily.

    Each subsequent variant will have, assigned to CPU, a superset of the
    CPU-assigned nodes in the previous variant.
    """
    nodes = sm.meta_graphs[0].graph_def.node
    tenth = len(nodes) // 10

    unassigned = set([x for x in range(len(nodes))])

    output_str = "saved_model.{}pct.cpu.pb"

    # assign 10, 20, 30, 40, 50% at random to CPU0
    for p in range(1, 6):
        sample = random.sample(unassigned, tenth)
        for i in sample:
            nodes[i].device = CPU0
            unassigned.remove(i)

        # save the model
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            print("Created output directory: {}".format(output_dir))
        output_filename = "saved_model.{}pct.cpu.pb".format(p*10)
        output = os.path.join(output_dir, output_filename)
        file_io.write_string_to_file(output,sm.SerializeToString())

        # echo a confirmation
        print("assigned {} new nodes to CPU,".format(len(sample)),
              "new model saved as {}".format(output_filename))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="directory to save the output models")
    parser.add_argument("-i", "--input_path", default=DEFAULT_MODEL_PATH,
                        help="path to find the saved_model file")
    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_arguments()

    sm = saved_model_pb2.SavedModel()

    pbpath = args.input_path
    if file_io.file_exists(pbpath):
        data = file_io.FileIO(pbpath, 'rb').read()
        sm.ParseFromString(data)
        # print_nodes(sm)
        rand_assign(sm, args.output_dir)
    else:
        print("No saved_model.pb found at".format(args.input_path))
        sys.exit()
