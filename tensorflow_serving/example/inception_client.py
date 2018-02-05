# Copyright 2015 Google Inc. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
from grpc.framework.interfaces.face.face import AbortionError
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import numpy as np
import pickle
import time
import threading


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp/imagenet/images/',
        'path to working directory')
tf.app.flags.DEFINE_integer('num_tests', 10, 'number of images to predict')
tf.app.flags.DEFINE_integer('concurrency', 1, 
        'number of concurrent prediction requests')
FLAGS = tf.app.flags.FLAGS


class _ResultCounter(object):
    """Counter for Inception prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = self._done = self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    """Create RPC callback function.
    
    Args:
        label: The correct label
        result_counter: Counter for prediction result
    """
    def _callback(result_future):
        """Callback function. Calculate statistics for the prediction result.
        
        TODO: Improve judgment heuristic.
        """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            response = "".join(np.array(
                result_future.result().outputs['scores'].string_val))
            hit = False
            for l in label.split(','):
                if l.strip() in response:
                    hit = True
                    break
            if not hit:
                result_counter.inc_error()
            result_counter.inc_done()
            result_counter.dec_active()

    return _callback


def do_inference(hostport, work_dir, concurrency, num_tests):
    """Test Inception prediction.

    Args:
      hostport: Host:port address of PredictionService.
      work_dir: The full path of working directory for test data set.
      concurrency: Max number of concurrent requests.
      num_test: Number of test images.
    """
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    counter = _ResultCounter(num_tests, concurrency)

    # load label map
    labelmap = {}
    with open(work_dir + "label_map", 'rb') as pickle_file:
        labelmap = pickle.load(pickle_file)

    images = labelmap.keys()
    for i in range(num_tests):
        # for now, recycle the images
        name = images[i%(len(images))]
        label = labelmap[name]
        with open(work_dir + name, 'rb') as f:
            data = f.read()
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'inception'
            request.model_spec.signature_name = 'predict_images'
            try:
                request.inputs['images'].CopyFrom(
                        tf.contrib.util.make_tensor_proto(data, shape=[1]))
                counter.throttle()
                result_future = stub.Predict.future(request, 10.0)  # 10 secs timeout
                result_future.add_done_callback(_create_rpc_callback(
                        label, counter))
            except:
                print("Error reading image {}".format(work_dir+name))
    return counter.get_error_rate()


def do_inference2(hostport, work_dir, num_tests):
    """Test Inception prediction.

    Args:
      hostport: Host:port address of PredictionService.
      work_dir: The full path of working directory for test data set.
      num_test: Number of test images.
    """
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    labelmap = {}
    with open(work_dir + "label_map", 'rb') as pickle_file:
        labelmap = pickle.load(pickle_file)
    images = labelmap.keys()
    for i in range(num_tests):
        name = images[i]
        label = labelmap[name]
        with open(work_dir + name, 'rb') as f:
            data = f.read()
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'inception'
            request.model_spec.signature_name = 'predict_images'
            try:
                request.inputs['images'].CopyFrom(
                        tf.contrib.util.make_tensor_proto(data, shape=[1]))
                result = stub.Predict(request, 10.0)  # 10 secs timeout
                scores = np.array(result.outputs['classes'].string_val)
                print("Prediction: {}\nLabel: {}".format(scores, label))
            except AbortionError as e:
                print("Error reading image {}".format(work_dir+name))
                print(e.details)


def main(_):
    if FLAGS.num_tests > 1000000000 or FLAGS.num_tests < 1:
        print("num_tests should be between 1 and 1B")
        return

    start_time = time.time()

    error_rate = do_inference(FLAGS.server, FLAGS.work_dir, 
            FLAGS.concurrency, FLAGS.num_tests)

    end_time = time.time()
    runtime = end_time - start_time
    print('\nResult:{},{},{},{}'.format(FLAGS.concurrency, FLAGS.num_tests,
            runtime, error_rate))


if __name__ == '__main__':
  tf.app.run()
