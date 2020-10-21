# run with
# sudo python3 image_parser.py vec2image
# or
# sudo python3 image_parser.py image2vec

import os
from cv2 import cv2
import sys
import numpy as np
import gensim.models.word2vec as w2v # pip3 install gensim
import pandas as pd # pip3 install pandas
import argparse 

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE # pip3 install sklearn
from sklearn.decomposition import PCA
from PIL import Image

# test_image = [[0.0, 20.0, 40.0, 60.0], [80.0, 100.0, 120.0, 140.0], [160.0, 180.0, 200.0, 220.0], [230.0, 240.0, 250.0, 255.0]]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def vec2image(name,output,size, words):
    # open model
    model = w2v.Word2Vec.load(os.path.join("trained", name))
    model.init_sims()

    # merge images
    canvas = np.zeros((5*size,5*size),dtype=np.uint8)
    words_file = open(words+".txt","r")
    x_axis = 0
    y_axis = 0
    for line in words_file:
        # get normalized word vector, range [-1, 1]
        norm_word = model.wv.word_vec(line.rstrip("\n"), use_norm=True)
        # add 1 to each element, new range [0, 2]
        norm_word = (norm_word + np.ones(size*size, dtype=int))
        # and multiply by 127.5, new range [0, 255]
        norm_word = (norm_word*(size*size-1)/2)
        # reshape to 16x16 img
        norm_word = norm_word.reshape(size,size)
        print(line.rstrip("\n"))
        print(norm_word)
        canvas[(x_axis*size):(x_axis*size+size),(y_axis*size):(y_axis*size+size)] = norm_word
        x_axis = x_axis + 1
        if x_axis == 5:
            x_axis = 0
            y_axis = y_axis+1
            
    # write image to file
    cv2.imwrite(output + ".png", canvas)

def image2vec(input,output):
    # read image
    im_array = cv2.imread(input + ".png")

    # convert image from rgb to grayscale
    gray = rgb2gray(im_array)

    # print resultant array
    print(gray)

    # save image
    cv2.imwrite(output + ".png", gray)

    # # set grayscale
    # plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
    # # remove axis
    # plt.axis("off")
    # # save image
    # plt.imsave("image2vec.png", im_array, cmap="gray")
    # # show image
    # plt.show()

def main():

    #Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model","-m", default = "corona.w2v", help = "Specify the model name (default = corona.w2v)")
    parser.add_argument("--command","-c", required = True, help = "Command to specify the mode (vec2image/image2vec)")
    parser.add_argument("--input", "-i", default = "vec2image", help = "Input file name for image2vec (default = vec2image)")
    parser.add_argument("--output","-o", required = True ,help = "Output file name")
    parser.add_argument("--size","-s", type = int , default = 16, help = "Size of the image generated (default = 16)")
    parser.add_argument("--words", "-w", default = "words", help = "Words file name to read (default = words)")
    opt = parser.parse_args()

    if opt.command == "vec2image":
        vec2image(opt.model,opt.output, opt.size, opt.words)
    elif opt.command == "image2vec":
        image2vec(opt.input,opt.output)

if __name__ == "__main__":
    main()