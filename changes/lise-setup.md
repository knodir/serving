## Initial setup

Order of operations:

install pip for python 2.7
```
sudo apt-get install python-pip python-dev
```

upgrade pip (maybe optional)
```
pip install --upgrade pip
pip install tensorflow
```

check TensorFlow installation directory:
```
python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
> /home/lise/.local/lib/python2.7/site-packages/tensorflow
```

running image recognition test
```
git clone https://github.com/tensorflow/models.git
cd models/tutorials/image/imagenet
python classify_image.py
```

install tensorflow-serving; requires su permissions
```
sudo pip install tensorflow-serving-api
```

install Bazel -- not sure if this step was necessary:  "only need if compiling source code"
```
sudo apt-get install openjdk-8-jdk
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel
```

install tensorflow-serving dependencies
```
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev
```

install model server
```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server
```

clone serving repo (dev branch) from knodir github
```
git clone --recurse_submodules https://github.com/knodir/serving.git
git checkout dev
```

copy model
```
python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist_model
```

install glog -- needed for timing, added by Nodir
```
sudo apt-get install glog
```

setup server
```
tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/
```

run tests
```
python tensorflow_serving/example/mnist_client.py --server=localhost:9000 --concurrency=1 --num_tests=1000
```

## Inception setup

**Note: the version of TF Serving in this repo is not building with the latest version of Bazel.**

Either install an older version of Bazel (0.9 works) or update Tensorflow to latest.

Build takes about 45 min total on LEAP-409.

```
bazel build -c opt tensorflow_serving/example/... --incompatible_load_argument_is_label=false
bazel build -c opt tensorflow_serving/model_servers:tensorflow_model_server
```

Download and decompress the inception model:

```
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
tar xzf inception-v3-2016-03-01.tar.gz
```

Export inception:

```
bazel-bin/tensorflow_serving/example/inception_saved_model --checkpoint_dir=inception-v3 --output_dir=/tmp/inception-export
```

**Note**: I received the warning when compiling on LEAP-409:

```
> Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
```

Set up the server:

```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception \
    --model_base_path=/tmp/inception-export &> inception_log 
```

Using the server (original inception client):

```
bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 --image=/path/to/jpeg
```

Using the server (modified inception client):
```
bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 --concurrency=1 --num_tests=100
```

## GPU

Sometimes Ubuntu has problems finding various CUDA libraries.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

Real time (well, every 0.1s) monitoring of GPU usage:
```
watch -n0.1 nvidia-smi
```


## Tools

Fetch 1000 images from imagenet, with no more than 10 per category.
Fetched images need to be checked for corruption.
```
python imgfetch.py -i 1000 -s 10
python cleanimages.py
```
(this process is horribly brittle and prone to problems)
