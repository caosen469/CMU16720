import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color

import sklearn.cluster as cluster


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    filter_scales = opts.filter_scales
    # print(filter_scales)
    # ----- TODO -----
    if len(img.shape) < 3:  # convert gery scale image to three channel
        img = np.stack([img, img, img], axis=-1)

    # Loop through the filters and scales to extract response
    img = skimage.color.rgb2lab(img)  # convert the image to the lab color space
    img_scale = []

    for each_scale in filter_scales:
        # convolution with each channel
        channel1 = scipy.ndimage.gaussian_filter(img[:, :, 0], each_scale)
        channel2 = scipy.ndimage.gaussian_filter(img[:, :, 1], each_scale)
        channel3 = scipy.ndimage.gaussian_filter(img[:, :, 2], each_scale)
        gaussian_img = np.stack([channel1, channel2, channel3], axis=-1)

        channel1 = scipy.ndimage.gaussian_laplace(img[:, :, 0], each_scale)
        channel2 = scipy.ndimage.gaussian_laplace(img[:, :, 1], each_scale)
        channel3 = scipy.ndimage.gaussian_laplace(img[:, :, 2], each_scale)
        gaussian_laplace_img = np.stack([channel1, channel2, channel3], axis=-1)

        # x derivatives:
        channel1 = scipy.ndimage.gaussian_filter(img[:, :, 0], (each_scale, each_scale), (0, 1))
        channel2 = scipy.ndimage.gaussian_filter(img[:, :, 1], (each_scale, each_scale), (0, 1))
        channel3 = scipy.ndimage.gaussian_filter(img[:, :, 2], (each_scale, each_scale), (0, 1))
        x_image = np.stack([channel1, channel2, channel3], axis=-1)

        # y derivatives
        channel1 = scipy.ndimage.gaussian_filter(img[:, :, 0], (each_scale, each_scale), (1, 0))
        channel2 = scipy.ndimage.gaussian_filter(img[:, :, 1], (each_scale, each_scale), (1, 0))
        channel3 = scipy.ndimage.gaussian_filter(img[:, :, 2], (each_scale, each_scale), (1, 0))
        y_image = np.stack([channel1, channel2, channel3], axis=-1)

        img_scale.append(np.concatenate([gaussian_img, gaussian_laplace_img, x_image, y_image], axis=2))

    # filter_responses = np.concatenate([img_scale[0], img_scale[1]], axis=-1)
    filter_responses = np.concatenate(img_scale, axis=-1)

    return filter_responses


def compute_dictionary_one_image(opts, img):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """

    # ----- TODO -----
    all_response = extract_filter_responses(opts, img)

    # respnseDic = np.empty((0, 3 * len(opts.filter_scales)))
    # Random extract filters' response
    alpha = opts.alpha

    # The filtered resonse

    dictionary = np.empty((0, all_response.shape[2]))

    for i in range(alpha):
        randx = np.random.randint(0, img.shape[0] - 1, 1)
        randy = np.random.randint(0, img.shape[1] - 1, 1)

        # randx = np.random.randint(0, img.shape[0] - 1, (img.shape[0], 1))
        # randy = np.random.randint(0, img.shape[1] - 1, (img.shape[1], 1))
        # print(filter_response.shape)
        one_feature = all_response[int(randx), int(randy), :].reshape((1, 12*len(opts.filter_scales)))
        dictionary = np.concatenate((dictionary, one_feature), axis=0)
        # print(one_feature)
        # print(filter_response.shape)
        # np.vstack((filter_response, one_feature))
    np.save(join(opts.out_dir, 'dictionary.npy'), dictionary)
    # print(dictionary)
    return dictionary


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()

    # ----- TODO -----
    img_path_instance = join(data_dir, train_files[0])
    img_instance = Image.open(img_path_instance)
    img_instance = np.array(img_instance).astype(np.float32) / 255
    response_instance = extract_filter_responses(opts, img_instance)

    filter_responses = np.empty((0, response_instance.shape[2]))

    # iterate through all images
    for each in train_files:
        img_path = join(data_dir, each)

        img = Image.open(img_path)
        img = np.array(img).astype(np.float32) / 255
        response = compute_dictionary_one_image(opts, img)
        filter_responses = np.concatenate((filter_responses, response), axis=0)

    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)
    kmeans = cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    # print(dictionary.shape)
    return dictionary


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    # ----- TODO -----
    all_response_oneImage = extract_filter_responses(opts, img)
    # print(all_response_oneImage[1,1,:].reshape(1,24))
    # wordmap = np.empty((img.shape[0], img.shape[1], 3))
    wordmap = np.empty((img.shape[0], img.shape[1]))
    # For one pixel:

    for i in range(all_response_oneImage.shape[0]):
        for j in range(all_response_oneImage.shape[1]):
            # distance = scipy.spatial.distance.cdist(all_response_oneImage[i, j, :].reshape(1, ), dictionary)
            # print(all_response_oneImage[i, j, :].shape)
            # print(dictionary.shape)
            distance = scipy.spatial.distance.cdist(all_response_oneImage[i, j, :].reshape(1, len(opts.filter_scales)*12), dictionary)
            closet = distance.argmin()
            wordmap[i, j] = closet
    return wordmap
