import csv
import os
import string
import nltk #pip3 install nltk
import multiprocessing
import sklearn.manifold #pip3 install sklearn
from nltk.corpus import stopwords
import gensim.models.word2vec as w2v #pip3 install gensim
import pandas as pd #pip3 install pandas
import seaborn as sns #pip3 install seaborn
import codecs

def ask2vec(model):
    word2vec = w2v.Word2Vec.load(os.path.join("trained", model))
    while True:
        texto = input("Enter command (exit to terminate): ")
        if texto == "exit":
            break
        else:
            print(eval("word2vec.wv." + texto))

def main():
    modelname = input("Enter model name: ")
    ask2vec(str(modelname))
    
#EXAMPLES:

#most_similar(positive=["homer"]))
#most_similar(positive=["man", "police"]))
#most_similar(positive=["woman", "bart"], negative=["man"])

if __name__ == "__main__":
    main()