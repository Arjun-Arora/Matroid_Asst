import sys
import cv2 
import os 
import re
import numpy as np
# import sklearn as sk
from tqdm import tqdm
import tensorflow as tf
# import tensorflow as tf
import matplotlib.pyplot as plt 
import random

import argparse

# ported from rcmalli: https://github.com/rcmalli/keras-vggface.git
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

print(tf.__version__)

def grab_data(data_dir = "./data"):
    reg_compile = re.compile(".*_F")
    female_paths = []
    male_paths = []

    loc_dirs = [os.path.join(data_dir,"combined/aligned"),os.path.join(data_dir,"combined/valid")]
    for loc_dir in loc_dirs:
        result_F = []
        result_M = []
        for dirpath,dirnames,filenames in os.walk(loc_dir):
            result_F = result_F + [dirname for dirname in dirnames if reg_compile.match(dirname)]
            result_M = result_M + [dirname for dirname in dirnames if not reg_compile.match(dirname)]
        for directory in result_F:
            current_directory = os.path.join(loc_dir,directory)
            for (dirpath, dirnames, filenames) in os.walk(current_directory):
                female_paths += [os.path.join(current_directory,filename) for filename in filenames]
        for directory in result_M:
            current_directory = os.path.join(loc_dir,directory)
            for (dirpath, dirnames, filenames) in os.walk(current_directory):
                male_paths += [os.path.join(current_directory,filename) for filename in filenames]
    return female_paths,male_paths


def createDataset(female_paths,male_paths,data_dir="./data"):
    m_0 = len(male_paths)
    m_1 = len(female_paths)

    male_feats = []
    female_feats = []
    male_dir,female_dir = (os.path.join(data_dir,'male_descriptors.npy'),
                          os.path.join(data_dir,'female_descriptors.npy'))
    layer_name = 'fc7'
    net =  VGGFace(model='vgg16')
    out = net.get_layer(layer_name).output
    net = Model(net.input, out)
    net.summary()
    if not os.path.exists(male_dir):
        for male_path in tqdm(male_paths):
            male_im = cv2.resize(plt.imread(male_path),(224,224)).astype(np.float32)[np.newaxis,:,:,:]
            male_im = utils.preprocess_input(male_im, version=1)

            male_feat = net.predict(male_im)
            # print(male_feat.shape)

            male_feats.append(male_feat)
        male_feats = np.concatenate(male_feats,axis=0)
        with open(male_dir,'wb') as f:
            np.save(f,male_feats)
    else:
        print("male descriptors exists, skipping...")
    if not os.path.exists(female_dir):
        for female_path in tqdm(female_paths):
            female_im = cv2.resize(plt.imread(female_path),(224,224)).astype(np.float32)[np.newaxis,:,:,:]
            female_im = utils.preprocess_input(female_im, version=1)

            female_feat = net.predict(female_im)
            # print(female_feat.shape)

            female_feats.append(female_feat)
        female_feats = np.concatenate(female_feats,axis=0)
        with open(female_dir,'wb') as f:
            np.save(f,female_feats)
    else:
        print("female descriptors exist, skipping...")
    return female_dir,male_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create feature vectors for model')
    parser.add_argument('--data_dir', default="./data", type=str, help='data directory')
    args = parser.parse_args()
    
    female_paths,male_paths = grab_data(args.data_dir)
    female_dir,male_dir = createDataset(female_paths,male_paths,args.data_dir)
    print("female descriptors located at: {}".format(female_dir))
    print("male descriptors located at: {}".format(male_dir))



