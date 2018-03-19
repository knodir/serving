#usr/bin/env python

import subprocess
import os
import time
import shutil

CLIENT = "bazel-bin/tensorflow_serving/example/inception_client"
TEST_SIZE = [100, 1000, 10000]
CONCURRENCY = [1, 2, 4, 6, 8, 10]

OUTPUT_PATH = "inception_random-{}pct"

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

def load_model(n):
    model_src = os.path.join("/tmp/inception-split-models", str(n))
    model_dest = os.path.join("/tmp/inception-export", str(n))
    shutil.copytree(model_src, model_dest)
    print("copied model {} to {}".format(model_src, model_dest))
    time.sleep(10)  # wait for server to detect model 


if __name__ == "__main__":
    benchmarks_dir = os.path.join(os.getcwd(), "inception-random-benchmarks")
    if not os.path.exists(benchmarks_dir):
        os.makedirs(benchmarks_dir)
    for i in range(1, 12):
        if i > 1:
            load_model(i)
        path = OUTPUT_PATH.format((i-1)*10)
        for c in CONCURRENCY:
            for n in TEST_SIZE:
                run_inference(n, c, path)
        clean_file(path, "Result:")
        shutil.move(path, os.path.join(benchmarks_dir, path))
