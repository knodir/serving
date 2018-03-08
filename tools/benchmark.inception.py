#!usr/bin/env python

import subprocess
import time

CLIENT = "bazel-bin/tensorflow_serving/example/inception_client"
TEST_SIZE = [100, 1000, 10000]
CONCURRENCY = [1, 2, 4, 6, 8, 10]

OUTPUT_PATH = "inception_{}"

def run_inference(n, c, path):
    cmd = "{} --server=localhost:9000 --num_tests={} --concurrency={} >> {}".format(
           CLIENT, n, c, path)
    print("Running Inception with num_tests={}, concurrency={}...".format(n, c))
    subprocess.call(cmd, shell=True)

def clean_file(path, prefix):
    lines = []
    with open(path, "rb") as f:
        lines = f.readlines()
    keep = []
    for line in lines:
        if line.startswith(prefix):
            keep.append(line[len(prefix):])
    with open(path, "wb") as f:
        f.writelines(keep)

if __name__ == "__main__":
    path = OUTPUT_PATH.format(time.strftime("%d-%m-%Y_%H-%M-%S"))
    for c in CONCURRENCY:
        for n in TEST_SIZE:
            run_inference(n, c, path)
    clean_file(path, "Result:")
