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
    result = np.histogram(wordmap, bins=[a for a in range(K + 1)], range=[0,K])

    # L1 Normalized
    return result[0] / result[0].sum()


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
    finest_layer_weight = 0.5

    # devide the word map into cells
    parts = pow(2, L)  # devide the image into 2^L parts
    width = wordmap.shape[0]
    height = wordmap.shape[1]
    partition_width = width // pow(2, L)
    partition_height = height // pow(2, L)

    partition_result = []
    partition_contented = np.empty((0, K))
    # print(partition_contented.shape)

    for row in range(pow(2, L)):
        for column in range(pow(2, L)):
            # print([row, column])
            # each partition
            partition = wordmap[row * partition_width:(row + 1) * partition_width,
                        column * partition_height:(column + 1) * partition_height]
            # calculate the histogram for that partition
            partition_histogram = get_feature_from_wordmap(opts, partition).reshape(1, K)
            # print(partition_histogram.shape)
            # contenate the histogram the result
            partition_contented = np.concatenate((partition_contented, partition_histogram), axis=0)

    partition_contented = partition_contented.reshape((1, partition_contented.shape[0] * partition_contented.shape[1]))
    # print((partition_contented / partition_contented.sum()).sum())

    # Normalized the histogram, this is the L1 histogram for the fines layer
    normalized_partition = partition_contented / partition_contented.sum()
    weighted_normalized_partition = finest_layer_weight * normalized_partition

    # Pyramid = normalized_partition

    # for each in reversed(range(L)):
    #     Layer_partion = np.empty((0,1))
    #     splited_partiton = np.split(normalized_partition, normalized_partition.shape[1]/4, axis=1)
    #
    #     for each in range(len(splited_partiton)):
    #         sum = splited_partiton[each].sum()
    #         Pyramid = np.append(Pyramid, np.array([[sum]]).reshape(1,1))
    # Pyramid = Pyramid.reshape(1, Pyramid.shape[0])
    # print(Pyramid)

    # return Pyramid

    Pyramid = weighted_normalized_partition
    # Other_Layers = np.empty((L, 0))
    count = 1
    Other_Layers = []

    for each in reversed(range(L)):
        This_layer = np.empty((1, 0))
        if each == 0:
            This_layer_weight = pow(2, -L)
        else:
            This_layer_weight = finest_layer_weight * pow(2, -count)
        # print(This_layer_weight)
        splited_partiton = np.split(normalized_partition, normalized_partition.shape[1] / pow(4, count), axis=1)
        count += 1

        for each in range(len(splited_partiton)):
            sum = splited_partiton[each].sum()

            This_layer = np.concatenate((This_layer, np.array([[sum]]).reshape(1, 1)), axis=1)

            Weighted_This_Layer = This_layer * This_layer_weight

        # Other_Layers = np.concatenate((Other_Layers, Weighted_This_Layer), axis=1)
        Other_Layers.append(Weighted_This_Layer)


    Other_Layers_np = np.empty((1,0))
    for i in reversed(range(len(Other_Layers))):
        Other_Layers_np = np.concatenate((Other_Layers_np, Other_Layers[i]), axis=1)
    Other_Layers_np = Other_Layers_np.flatten()
    Other_Layers_np = Other_Layers_np.reshape((1, Other_Layers_np.shape[0]))

    # print(Other_Layers.shape)
    Pyramid = np.concatenate((Pyramid, Other_Layers_np), axis=1)
    hist_all = Pyramid.reshape((Pyramid.shape[1]))

    return hist_all


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
    # Build the image
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255

    # Get the wordmap
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    return feature


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
    # # generate the dictionary
    # dictionary = visual_words.compute_dictionary(opts)

    # Create features
    first_image_path = join(data_dir, train_files[0])
    len_features = get_image_feature(opts, first_image_path, dictionary).shape[0]
    features = np.zeros((0, len_features))
    # for each in train_files:
    #     img_path = join(data_dir, each)
    #     feature = get_image_feature(opts, img_path, dictionary)
    #     feature = feature.reshape(1, len_features)
    #     features = np.concatenate((features, feature), axis=0)
    global get_image_feature_one
    def get_image_feature_one(image_path):
        global features
        img_path = join(data_dir, image_path)
        feature = get_image_feature(opts, img_path, dictionary)
        feature = feature.reshape(1, len_features)
        features = np.concatenate((features, feature), axis=0)
        return

    if __name__ == "__main__":
        pool = multiprocessing.Pool(processes=n_worker)
        print(type(train_files))
        pool.map(get_image_feature_one, train_files)

    # Creating labels
    labels = train_labels

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
                        features=features,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num,
                        )


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
    # compute the minimum for each row
    minimum_bin = np.minimum(histograms, word_hist)
    score_eachImgae = np.sum(minimum_bin, axis=1)
    sim = 1 - score_eachImgae
    return sim


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
    # Set an empty confusion matrix
    conf = np.zeros((8, 8))
    # Set the accuracy
    accuracy = 0


    # Load the training data and the label

    count = 0
    # for one_test_path in test_files:
    #     test_image_path = join(data_dir, one_test_path)
    #     features = get_image_feature(opts, test_image_path, dictionary)
    #     # The distance between test image and each trained image
    #     distance = distance_to_set(features, trained_system['features'])
    #     # The shortest distance trained image
    #     predict_type_position = np.argmin(distance)
    #     # predicted result, which is a number from 0 to 7
    #     predict_result = trained_system['labels'][predict_type_position]
    #
    #     # add it to the confusion matrix
    #     j = predict_result
    #     i = test_labels[count]
    #     conf[i, j] += 1
    #
    #     if predict_result == test_labels[count]:
    #         accuracy += 1
    #     count += 1
    global evaluate_one_image
    def evaluate_one_image(one_test_path):
        global count
        global accuracy
        test_image_path = join(data_dir, one_test_path)
        features = get_image_feature(opts, test_image_path, dictionary)
        # The distance between test image and each trained image
        distance = distance_to_set(features, trained_system['features'])
        # The shortest distance trained image
        predict_type_position = np.argmin(distance)
        # predicted result, which is a number from 0 to 7
        predict_result = trained_system['labels'][predict_type_position]

        # add it to the confusion matrix
        j = predict_result
        i = test_labels[count]
        conf[i, j] += 1
        if predict_result == test_labels[count]:
            accuracy += 1
        count += 1

    print(__name__)
    # if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=n_worker)
    res = pool.map(evaluate_one_image, test_files)
    print(res)


    accuracy = accuracy / test_labels.shape[0]

    return [conf, accuracy]

