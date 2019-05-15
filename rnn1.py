from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.models import load_model
from keras.models import Sequential
from gensim.models import KeyedVectors
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from keras.optimizers import Adam
from keras.utils import Sequence
from IPython.display import HTML
from itertools import chain
from keras.utils import plot_model
import numpy as np
import pandas as pd
import random
import json
import re
from keras.models import model_from_json
RANDOM_STATE = 50
TRAIN_FRACTION = 0.7
def create_train_valid(features,
                       labels,
                       num_words,
                       train_fraction=0.7):
    """Create training and validation features and labels."""
    
    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    # Memory management
    import gc
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid

def get_embeddings(model):
    """Retrieve the embeddings in a model"""
    embeddings = model.get_layer(index = 0)
    embeddings = embeddings.get_weights()[0]
    embeddings = embeddings / np.linalg.norm(embeddings, axis = 1).reshape((-1, 1))
    embeddings = np.nan_to_num(embeddings)
    return embeddings

def print1(seed):
    for i in seed:
        print(idx_word.get(i))

def generate_new(model,sequences,idx_word,seed_length=50,new_words=10,diversity=1,return_output=False,n_gen=1):
    seq = random.choice(sequences)
    seed_idx = random.randint(0, len(seq) - seed_length - 10)
    end_idx = seed_idx + seed_length
    seed = seq[seed_idx:end_idx]
    #print1(seed)
    generated = seed[:] + ['#']
    actual = generated[:] + seq[end_idx:end_idx + new_words]
    #print1(actual)
    for i in range (new_words):
        preds = model.predict(np.array(seed).reshape(1,-1))[0].astype(np.float64)
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)[0]
        next_idx = np.argmax(probas)
        #print(idx_word.get(next_idx))
        seed=seed[1:]
        seed += [next_idx]
        generated.append(next_idx)
    n = []
    k = []
    #print1(generated)
    for i in generated:
        n.append(idx_word.get(i, '< --- >'))
    for i  in actual:
        k.append(idx_word.get(i,'< --- >'))
    return n,k

#one hot encoding
"""
label_array= np.zeros((len(features),num_words),dtype=np.int8)

for index,label_name in enumerate(labels):
    label_array[index][label_name]=1
"""
dataset = pd.read_csv('rnn1.csv')
dataset=dataset['patent_abstract'].tolist()
#dataset=dataset[:10]
tokenizer=Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=False,split=' ')
tokenizer.fit_on_texts(dataset)
sequences=tokenizer.texts_to_sequences(dataset)
sentences=tokenizer.index_word
idx_word=sentences
features = []
labels = []
training_length = 50
num_words=len(sentences)+1
# Iterate through the sequences of tokens
for seq in sequences:
    #print(len(seq))
    # Create multiple training examples from each sequence
    for i in range(training_length, len(seq)):
        # Extract the features and label
        extract = seq[i - training_length:i + 1]
        features.append(extract[:-1])
        labels.append(extract[-1])
             
X_train, X_valid, y_train, y_valid = create_train_valid(features, labels, num_words)
training_dict = {'X_train': X_train, 'X_valid': X_valid, 
                     'y_train': y_train, 'y_valid': y_valid}
#from here embeddings

glove_vectors = 'glove.6B.100d.txt'
glove = np.loadtxt(glove_vectors, dtype='str', comments=None, encoding='UTF8')

# Extract the vectors and words
vectors = glove[:, 1:].astype('float')
words = glove[:, 0]

# Create lookup of words to vectors
word_lookup = {word: vector for word, vector in zip(words, vectors)}
# New matrix to hold word embeddings
embedding_matrix = np.zeros((num_words, vectors.shape[1]))

for i, word in enumerate(sentences.keys()):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector

#to here
        
        
#model creation
model = Sequential()

# Embedding layer
model.add(
    Embedding(input_dim=num_words,
              input_length = training_length,
              output_dim=100,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_words, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.save('my_model.h5')

# Load in model and demonstrate training
model = load_model('my_model.h5')
model.summary()

h = model.fit(training_dict['X_train'], training_dict['y_train'], epochs = 5, batch_size = 2048, 
          validation_data = (training_dict['X_valid'], training_dict['y_valid']), 
          verbose = 1)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
#model over

#testing codes



json_file = open('own_embeddings_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_new_weights.h5")
loaded_model.save('model_new.hdf5')
loaded_model=load_model('model_new.hdf5')

a,b = generate_new(model,sequences,idx_word,seed_length=50,new_words=20,diversity=1)
str1=' '.join(a)
str2=' '.join(b)
print("actual :")
print(str2)
print("Generated :")
print(str1)
