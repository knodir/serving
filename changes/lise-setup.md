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
git clone https://github.com/knodir/serving.git
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

run tests
```
python tensorflow_serving/example/mnist_client.py --server=localhost:9000 --concurrency=1 --num_tests=1000
```

