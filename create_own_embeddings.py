import nltk
nltk.download()
from nltk.corpus import brown
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
dataset = pd.read_csv('rnn1.csv')
dataset=dataset['patent_abstract'].tolist()
list_new=[]
for i in range (len(dataset)):
    list_new.append(dataset[i].split())

w2v=Word2Vec(list_new,size=100,window=5,min_count=5,negative=15,iter=10,workers=1)
""" size is dimension of embedding vector 
    window is how many neighbour words are we considering to get semantics of words
    """
w2v.wv.most_similar(positive=["Neuron"])
w2v.wv.similarity("Neural","Artificial")

"""now we will create embedding matrix"""
tokenizer=Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=False,split=' ')
tokenizer.fit_on_texts(dataset)
sequences=tokenizer.texts_to_sequences(dataset)
sentences=tokenizer.index_word
idx_word=sentences
embedding_matrix = np.zeros((len(tokenizer.word_index),100))
for i, word in idx_word.items():
    try:
        embedding_vector = w2v[word]
    except:
        pass
    try:
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    except:
           pass
     