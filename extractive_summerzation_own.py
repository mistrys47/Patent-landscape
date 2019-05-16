def check(s):
    import string
    table = str.maketrans('', '', string.punctuation)
    s=s.split()
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    for i in range(len(s)):
        s[i]=s[i].translate(table)
        s[i]=s[i].lower()
        if s[i].isalpha()==False:
            s[i]=' '
        if s[i] in stop_words:
            s[i]=' '
    return ' '.join(s)
        



bq_project = 'fast-rune-240311'
print(bq_project)


import tensorflow as tf
import numpy as np
import pandas as pd
import os

seed_name = 'video_codec'
seed_file = 'seeds/video_codec.seed.csv'
patent_dataset = 'patents-public-data:patents.publications_latest'
num_anti_seed_patents = 15000
if bq_project == '':
    raise Exception('You must enter a bq_project above for this code to run.')
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/Deep Mistry/patents-public-data/patents-landscape-007ab831042b.json"
print('Credendtials from environ: {}'.format(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))

import expansion

expander = expansion.PatentLandscapeExpander(
    seed_file,
    seed_name,
    bq_project=bq_project,
    patent_dataset=patent_dataset,
    num_antiseed=num_anti_seed_patents)


training_data_full_df, seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = \
    expander.load_from_disk_or_do_expansion()

training_df = training_data_full_df[
    ['publication_number', 'title_text', 'abstract_text', 'claims_text', 'description_text', 'ExpansionLevel', 'refs', 'cpcs']]
list(training_data_full_df.columns.values)

from nltk.tokenize import sent_tokenize
sentences = []
for s in training_df['abstract_text']:
    x=sent_tokenize(s)
    for i in range(len(x)):
        x[i]=check(x[i])
    sentences.extend(x)
#checkpoint REAL
from word2vec import Word2Vec
word2vec5_9m = Word2Vec('5.9m')
w2v_runtime = word2vec5_9m.restore_runtime()

#preparing embeddings for
word_embeddings = {}
for i in range(len(sentences)):
    words=sentences[i].split()
    for word in words:
        word_embeddings[word]=w2v_runtime.load_embedding(word)

sentence_vectors = []
for i in sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((300,))
  sentence_vectors.append(v)

# similarity matrix 
  # in similarity matrix we will have to use full length of sentences but because of memory error we have used half of it
  # in real application it won't happen
  #real code is in comment
"""
sim_mat = np.zeros([int(len(sentences)), int(len(sentences))])
from sklearn.metrics.pairwise import cosine_similarity
for i in range(int(len(sentences))):
  for j in range(int(len(sentences))):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i], sentence_vectors[j])[0,0]"""
sim_mat = np.zeros([int(len(sentences)/2), int(len(sentences)/2)])
from sklearn.metrics.pairwise import cosine_similarity
for i in range(int(len(sentences)/2)):
  for j in range(int(len(sentences)/2)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i], sentence_vectors[j])[0,0]

import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],sentences[i]) for i in range(19827)), reverse=True)

n=input("Enter no of summary sentences you want : ")
n=int(n)
for i in range(n):
  print(i+1)
  print(ranked_sentences[i][1])
  print()


#for real example load individual sentences into sentences then run code from checkpoint REAL


