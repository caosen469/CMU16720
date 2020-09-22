import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    # print(np.histogram(wordmap)[0].shape)
    result = np.histogram(wordmap, bins=[a for a in range(K+1)])
    result = np.histogram(wordmap,)

    # L1 Normalized
    return result[0]/result[0].sum()



def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    K = opts.K
    L = opts.L
    # ----- TODO -----

    # compute the histogram of the finest layer
    # weights = pow(2, -L), weight for the last layer is always 1/2
    weight = 0.5

    # devide the word map into cells
    parts = pow(2, L) # devide the image into 2^L parts
    width = wordmap.shape[0]
    height = wordmap.shape[1]
    partition_width = width // pow(2,L)
    partition_height = height // pow(2, L)

    partition_result = []
    partition_contented = np.empty((0, 10))
    # print(partition_contented.shape)

    for row in range(pow(2,L)):
        for column in range(pow(2, L)):
            # print([row, column])
            # each partition
            partition = wordmap[row*partition_width:(row+1)*partition_width, column*partition_height:(column+1)*partition_height]
            # calculate the histogram for that partition
            partition_histogram = get_feature_from_wordmap(opts, partition).reshape(1,K)
            # print(partition_histogram.shape)
            # contenate the histogram the result
            partition_contented = np.concatenate((partition_contented, partition_histogram), axis=0)

    partition_contented = partition_contented.reshape((1, partition_contented.shape[0]*partition_contented.shape[1]))
    # print((partition_contented / partition_contented.sum()).sum())

    # Normalized the histogram, this is the L1 histogram for the fines layer
    normalized_partition = partition_contented / partition_contented.sum()

    Pyramid = normalized_partition

    for each in reversed(range(L)):




    return normalized_partition


def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    pass

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    pass

    ## example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    pass    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    pass

