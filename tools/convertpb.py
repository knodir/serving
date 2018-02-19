# Convert between a saved_model.pb and saved_model.pbtxt file
#
# usage: python convertpb.py /saved_model/src/dir/ /output/dir/
#        (default output directory is the current directory)

from __future__ import print_function

import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants

from google.protobuf import text_format

import os
import sys

def confirm(output_path):
    if file_io.file_exists(output_path):
        confirm = raw_input("File exists at {}. Overwrite? y/n: ".format(
            output_path))
        return confirm in 'yesYes'
    return True

def txt_to_pb(output, sm, data):
    if confirm(output):
        text_format.Merge(data.decode('utf-8'), sm)
        file_io.write_string_to_file(output, sm.SerializeToString())

def pb_to_txt(output, sm, data):
    if confirm(output):
        sm.ParseFromString(data)
        file_io.write_string_to_file(output, str(sm))


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print("Usage: python convertpb.py /saved_model/src/dir/",
                "[/output/dir/]")
        sys.exit()

    src_dir = args[1]

    sm = saved_model_pb2.SavedModel()
    output_dir = os.getcwd() if len(args) < 3 else args[2]

    pbpath = os.path.join(src_dir, constants.SAVED_MODEL_FILENAME_PB)
    txtpath = os.path.join(src_dir, constants.SAVED_MODEL_FILENAME_PBTXT)
    if file_io.file_exists(pbpath):
        data = file_io.FileIO(pbpath, 'rb').read()
        output_path = os.path.join(output_dir, 
                constants.SAVED_MODEL_FILENAME_PBTXT)
        pb_to_txt(output_path, sm, data)
    elif file_io.file_exists(txtpath):
        data = file_io.FileIO(txtpath, 'rb').read()
        output_path = os.path.join(output_dir, 
                constants.SAVED_MODEL_FILENAME_PB)
        txt_to_pb(output_path, sm, data)
    else:
        print("No saved_model file found.")
        sys.exit()
