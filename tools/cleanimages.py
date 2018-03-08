#!/usr/bin/env python2.7

from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import os
import pickle

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                                   'PredictionService host:port')
tf.app.flags.DEFINE_string('dir', '/tmp/imagenet/images', 'path to images')
FLAGS = tf.app.flags.FLAGS


def main(_):
    # get list of images
    images = {}
    labelpath = os.path.join(FLAGS.dir, "labelmap")
    with open(labelpath, 'rb') as f:
        images = pickle.load(f)

    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    for name in images.keys():
        imgpath = os.path.join(FLAGS.dir, name)
        # Send request
        try:
            with open(imgpath, 'rb') as f:
                data = f.read()
                request = predict_pb2.PredictRequest()
                request.model_spec.name = 'inception'
                request.model_spec.signature_name = 'predict_images'
                request.inputs['images'].CopyFrom(
                    tf.contrib.util.make_tensor_proto(data, shape=[1]))
                result = stub.Predict(request, 10.0)  # 10 secs timeout
        except:
            if os.path.exists(imgpath):
                os.remove(imgpath)
            del images[name]
            print("Removed corrupted image {}".format(name))

    with open(labelpath, 'w+') as f:
        pickle.dump(images, f)


if __name__ == '__main__':
    tf.app.run()
