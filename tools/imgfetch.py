"""
Pull images from the ImageNet dataset and save them to disk, along with
pickle to get img mappings.

Usage:
    python imgfetch.py -i 1000 -s 10
    Fetches 1000 images, maximum of 10 images per label

TODO:
    add ability to switch directories
"""
#!usr/bin/env python2.7

from __future__ import print_function

import argparse
import os
import pickle
import shutil
import requests

from tensorflow.contrib.util import make_tensor_proto

SYNSET_BASE_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
LABELS = "tensorflow_serving/example/imagenet_metadata.txt"
IMAGE_DIR = "/tmp/imagenet/images/"

def get_images(lbl_path, img_path, num_images, synset_size):
    """Fetch images from image-net and map to labels.

    Return a map of images names to label.
    """
    if not os.path.exists(lbl_path):
        print("No synset label file found")
        return

    if os.path.exists(img_path):
        print("Directory is already in use")
        return

    os.makedirs(img_path)

    labelmap = {}
    with open(lbl_path, "r") as f:
        for line in f.readlines():
            synset, label = line.strip().split("\t", 1)
            imgs = get_synset_images(synset, img_path, synset_size)
            for i in imgs:
                labelmap[i] = label
            if len(labelmap) >= num_images:
                break

    with open(img_path + "labelmap", 'w+') as f:
        pickle.dump(labelmap, f)

    print("Downloaded {} images".format(len(labelmap)))

def get_synset_images(synset, directory, limit):
    """Fetch images associated with a specific synset and save to directory.
    Maximum number of images fetched is limit. Fetch all images if limit is 0.

    Return a list of all the images fetched.
    """
    imgs = []
    try:
        r = requests.get(SYNSET_BASE_URL + synset, timeout=8)
        for url in r.text.split('\r\n'):
            if len(url) == 0:
                continue
            name = url.split('/')[-1]
            # drop empty strings, or if file with that name exists
            if len(name) == 0 or os.path.exists(directory + name):
                continue
            try:
                img = requests.get(url, timeout=0.5, allow_redirects=False)
                if img.status_code == requests.codes.ok and validate(img.content):
                    with open(directory + name, 'wb') as out_file:
                        out_file.write(img.content)
                        imgs.append(name)
                        if len(imgs) == limit:
                            return imgs
            except requests.exceptions.RequestException:
                continue 
            except requests.exceptions.ConnectTimeout:
                continue
    except requests.exceptions.ReadTimeout:
        print("Timed out while fetching synset URLS")
    return imgs

def validate(img_data):
    """Check that the image passed is usable.
     Sometimes images in the synset are corrupt.
    """
    try:
        make_tensor_proto(img_data, shape=[1])
    except Exception as e:
        return False
    return True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int, default=10,
            help="Number of images to fetch")
    parser.add_argument("-s", type=int, default=3,
            help="Max number of images per synset")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    get_images(os.path.join(os.getcwd(), LABELS), IMAGE_DIR, args.i, args.s)
