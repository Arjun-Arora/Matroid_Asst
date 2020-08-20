import sys
import cv2 
import os 
import re
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import random
import argparse
# print(tf.__version__)

class descriptorGenderClassifier(tf.keras.Model):
    def __init__(self):
        super(descriptorGenderClassifier, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1000,input_shape=(None,1), activation=tf.nn.leaky_relu,name='D0')
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu,name='D1')
        self.dense3 = tf.keras.layers.Dense(10, activation=tf.nn.leaky_relu,name='D2')
        self.dense_end = tf.keras.layers.Dense(1, activation='sigmoid',name='DL')
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        out = self.dense_end(x)
        return out

model = descriptorGenderClassifier()

def main(args):

    data_dir = args.data_dir
    batch_size = args.batch_size
    train_prop = args.train_prop
    epochs = args.epochs
    model_dir = args.model_save_dir

    test_prop = val_prop = (1 - train_prop) / 2 
    # load data from descriptors
    female_descriptors = np.load(os.path.join(data_dir,"female_descriptors.npy"))
    male_descriptors = np.load(os.path.join(data_dir,"male_descriptors.npy"))

    # concatenate full dataset and labels
    m_len = male_descriptors.shape[0]
    f_len = female_descriptors.shape[0]
    dataset_descriptors = np.concatenate([male_descriptors,female_descriptors],axis=0)

    # males, zero, Women, one
    dataset_labels = np.concatenate([np.zeros((m_len,1)),np.ones((f_len,1))],axis=0)
    input_shape = dataset_descriptors.shape[1]
    # print("Dataset input shape: {}".format(dataset_descriptors.shape))
    # print("Dataset labels shape: {}".format(dataset_labels.shape))

    #create dataset
    dataset = tf.data.Dataset.from_tensor_slices((dataset_descriptors, dataset_labels))
    dataset = dataset.shuffle(m_len + f_len)
    dataset = dataset.batch(batch_size, drop_remainder = False)
    dataset_size = len(list(dataset))

    train_size = int(train_prop * dataset_size)
    val_size = int(val_prop * dataset_size)
    test_size = int(test_prop * dataset_size)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)



    model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.fit(train_dataset, epochs=epochs,validation_data=val_dataset)

    Test_X = []
    Test_Y = []
    for X,Y in list(test_dataset):
        Test_Y.append(Y.numpy())
        Test_X.append(X.numpy())
    Test_X = np.concatenate(Test_X)
    Test_Y = np.concatenate(Test_Y)
    y_pred = model.predict(Test_X) > 0.5
    print("\nclassification accuracy overall on Test set: {}\n".format(sk.metrics.accuracy_score(Test_Y, y_pred)))
    print("classification metrics per class on Test set: ")
    print(classification_report(Test_Y, y_pred,target_names=['male','female']))
    if model_dir is not None:
        model_path = os.path.join(model_dir,"descriptorGenderClassifier")
        model.save(model_path)
        print("saved model to: {}".format(model_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train,evaluate,and save model')
    parser.add_argument('--data_dir', default="./data", type=str, help='data directory')
    parser.add_argument('--batch_size',default=16,type=int,help='batch size for training')
    parser.add_argument('--train_prop',default=0.70,type=float,help='what proportion of model is for training (test and val split rest)')
    parser.add_argument('--epochs',default=10,type=int,help='number of epochs to train model')
    parser.add_argument('--model_save_dir',default=None,type=str,help='where to save descriptorGenderClassifier model')
    args = parser.parse_args()
    main(args)