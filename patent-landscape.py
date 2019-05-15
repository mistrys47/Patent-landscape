#you need to have whole github repo's files to run this code
#drive link:  https://github.com/google/patents-public-data/blob/master/models/landscaping/README.md
#first get data into training_df then run codes of wordcloud

bq_project = 'fast-rune-240311'
print(bq_project)


import tensorflow as tf
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




from word2vec import W2VModelDownload

model_name = '5.9m'
model_download = W2VModelDownload(bq_project)
model_download.download_w2v_model('patent_landscapes', model_name)
print('Done downloading model {}!'.format(model_name))


from word2vec import Word2Vec

word2vec5_9m = Word2Vec('5.9m')
w2v_runtime = word2vec5_9m.restore_runtime()


word="networks"
len(w2v_runtime.load_embedding(word))

w2v_runtime.find_similar('network', 10)


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





print('Seed/Positive examples:')
print(training_df[training_df.ExpansionLevel == 'Seed'].count())

print('\n\nAnti-Seed/Negative examples:')
print(training_df[training_df.ExpansionLevel == 'AntiSeed'].count())





import train_data
import tokenizer

# TODO: persist this tokenization data too
td = train_data.LandscapeTrainingDataUtil(training_df, w2v_runtime)
td.prepare_training_data(
    training_df.ExpansionLevel,
    training_df.abstract_text,
    training_df.refs,
    training_df.cpcs,
    0.8,
    50000,
    500)

#pre-processing

for i in range (len(training_df)):
    

#word cloud for abstracts
from wordcloud import WordCloud
import matplotlib.pyplot as plt
x11=[]

for i in range (len(training_df)):
    x11=x11+training_df['abstract_text'][i].lower().split()
spam_words = ' '.join(x11)
#print(spam_words)
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
spam_wc.to_file('wc.png')

#word cloud for refs
from wordcloud import WordCloud
import matplotlib.pyplot as plt
x11=[]
for i in range (len(training_df)):
    x11=x11+training_df['refs'][i].lower().split(',')
b11=[[x,x11.count(x)] for x in set(x11)]
d = {}
for i in range(len(b11)):
    d[b11[i][0]] = b11[i][1]
print(d)
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file('refs_wc.png')

#code 2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
x11=[]
for i in range (len(training_df)):
    x11=x11+training_df['refs'][i].lower().split(',')
#b11=[[x,x11.count(x)] for x in set(x11)]
#d = {}
#for i in range(len(b11)):
#    d[b11[i][0]] = b11[i][1]
#print(d)
d={ x:x11.count(x) for x in x11 }
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file('refs_wc.png')



