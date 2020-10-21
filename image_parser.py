# run with
# sudo python3 image_parser.py vec2image
# or
# sudo python3 image_parser.py image2vec

import os
import cv2
import sys
import numpy as np
import gensim.models.word2vec as w2v # pip3 install gensim
import pandas as pd # pip3 install pandas

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE # pip3 install sklearn
from sklearn.decomposition import PCA
from PIL import Image

# test_image = [[0.0, 20.0, 40.0, 60.0], [80.0, 100.0, 120.0, 140.0], [160.0, 180.0, 200.0, 220.0], [230.0, 240.0, 250.0, 255.0]]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def vec2image(name):
    # open model
    model = w2v.Word2Vec.load(os.path.join("trained", name))
    model.init_sims()

    # get normalized word vector, range [-1, 1]
    norm_word = model.wv.word_vec('generation', use_norm=True)

    # add 1 to each element, new range [0, 2]
    norm_word = (norm_word + np.ones(256, dtype=int))
    # and multiply by 127.5, new range [0, 255]
    norm_word = (norm_word*127.5)
    # reshape to 16x16 img
    norm_word = norm_word.reshape(16,16)
    
    # print vector
    print(norm_word)

    # write image to file
    cv2.imwrite("vec2image.png", norm_word)

def image2vec():
    # read image
    im_array = cv2.imread("vec2image.png")

    # convert image from rgb to grayscale
    gray = rgb2gray(im_array)

    # print resultant array
    print(gray)

    # save image
    cv2.imwrite("image2vec.png", gray)

    # # set grayscale
    # plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    # # remove axis
    # plt.axis('off')
    # # save image
    # plt.imsave("image2vec.png", im_array, cmap='gray')
    # # show image
    # plt.show()

def main():
    if(len(sys.argv) < 2):
        print("Missing arguments")
        return

    cmd = sys.argv[1]
    model_name = 'corona.w2v'

    if cmd == "vec2image":
        vec2image(model_name)
    elif cmd == "image2vec":
        image2vec()

if __name__ == "__main__":
    main()