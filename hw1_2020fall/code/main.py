from os.path import join

import numpy as np
from PIL import Image

import util
import visual_words
import visual_recog
from opts import get_opts

import sklearn.cluster as cluster

def main():
    opts = get_opts()

    # Q1.1
    # opts.filter_scales = [0.5, 1, 1.5, 2, 2.5]
    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)

    # Apply on aquarium/sun_aztvjgubyrgvirup.jpg
    # opts.filter_scales = [1, 2, 3]
    img_path2 = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    img2 = Image.open(img_path2)
    # img2 = np.array(img2).astype(np.float32)/255
    # filter_responses = visual_words.extract_filter_responses(opts, img2)
    # util.display_filter_responses(opts, filter_responses)
    #visual_words.compute_dictionary_one_image(opts, img)

    ## Q1.2
    # n_cpu = util.get_num_CPU()
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)
    # visual_words.compute_dictionary(opts)

    # Q1.3
    img_path = join(opts.data_dir, 'waterfall/sun_aastyysdvtnkdcvt.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)





    ## Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu-1)
    # visual_recog.build_recognition_system(opts)

    ## Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu-1)
    #conf, accuracy = visual_recog.evaluate_recognition_system(opts)

    #
    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
