# run with
# sudo python3 image_parser.py vec2image
# or
# sudo python3 image_parser.py image2vec

import csv
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def vec2image(name, output, size, words):
    # open model
    model = w2v.Word2Vec.load(os.path.join('../word2vec/trained', name))
    model.init_sims(replace=True)


    #words_file = open(words+".txt","r")

    with open('../word2vec/corpus.csv', newline='') as csvfile:
        row_reader = csv.reader(csvfile, delimiter = ' ')
        i=0
        for row in row_reader:
            # merge images
            canvas = np.zeros((8*size,8*size),dtype=np.uint8)
            x_axis = 0
            y_axis = 0
            for word in row:
                try:
                    # get normalized word vector, range [-1, 1]
                    norm_word = model.wv.word_vec(word.rstrip("\n"), use_norm=True)
                    # add 1 to each element, new range [0, 2]
                    norm_word = (norm_word + np.ones(size*size, dtype=int))
                    # and multiply by 127.5, new range [0, 255]
                    norm_word = (norm_word*(size*size-1)/2)
                    # reshape to 16x16 img
                    norm_word = norm_word.reshape(size,size)
                    canvas[(x_axis*size):(x_axis*size+size),(y_axis*size):(y_axis*size+size)] = norm_word
                    y_axis = y_axis + 1
                    if y_axis == 8:
                        y_axis = 0
                        x_axis = x_axis+1
                except:
                    continue
            # write image to file
            cv2.imwrite("./generated/" + output + str(i) + ".png", canvas)
            i+=1
            


def image2vec(name, input, output, size):

    model = w2v.Word2Vec.load(os.path.join('../word2vec/trained', name))

    # read image
    im_array = cv2.imread('./generated/' + input + ".png")

    # convert image from rgb to grayscale
    
    gray = rgb2gray(im_array)

    x_axis = 0
    y_axis = 0
    final_tweet = ""

    for i in range(64):
        norm_word = gray[(x_axis*size):(x_axis*size+size),(y_axis*size):(y_axis*size+size)]
        y_axis = y_axis + 1
        if y_axis == 8:
            y_axis = 0
            x_axis = x_axis+1

        #reshape to an array
        norm_word = norm_word.reshape(1,256)

        #denormalize
        norm_word = norm_word/127.5

        norm_word = norm_word - np.ones(256, dtype=int)

        if np.average(norm_word) == -1.0:
            break
        else:
            #search in model
            word = model.wv.most_similar(norm_word)
            final_tweet += " " + (word[0][0])

    #print result
    print(f"The obtained tweet is:{final_tweet}")

        #save image
        #cv2.imwrite(output + ".png", norm_word)

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
        image2vec(opt.model, opt.input,opt.output, opt.size)

if __name__ == "__main__":
    main()