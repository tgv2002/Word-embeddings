import re
import numpy as np
import json
from math import sqrt, floor, ceil
import nltk
from scipy.sparse import csr_matrix
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD, PCA
from random import sample, choices
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from keras.models import Model
from keras.layers.core import Dense, Reshape
from keras.layers import Input, Lambda, Dense, dot, concatenate
from keras.layers.embeddings import Embedding
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
nltk.download('punkt')

from gensim.models import KeyedVectors


EMBEDDING_DIM = 200
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 3
# FOLDER_PATH = '/content/gdrive/My Drive/Nlp-assign2/'
FOLDER_PATH = './'

def get_vocabulary(sentences, threshold=5):
    word_frequencies = {}
    for sentence in sentences:
        words = tokenize_text(sentence)
        for word in words:
            if word_frequencies.get(word) is not None:
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1
    return {w: v for w, v in word_frequencies.items() if v >= threshold}

def tokenize_text(text):
    words = re.split("\||/|\=|_|\:|\+|,|~|\^|#|\[|\]|\(|\)|\{|\}|<|>|\!|&|;|\?|\*|%|\$|@|`| ", text.lower())
    words = [word.strip("|/=.-_:'+,~^#[](){}<>!&;?*%$@`\"") for word in words if len(word) > 1]
    start_pad = ['<S>' for _ in range(WINDOW_SIZE - 1)]
    end_pad = ['<E>' for _ in range(WINDOW_SIZE - 1)]
    return start_pad + words + end_pad

def update_sentences(vocab, sentences):
    unk_c = 0
    for i in range(len(sentences)):
        words = tokenize_text(sentences[i])
        for j in range(len(words)):
            if words[j] not in vocab:
                unk_c += 1
                words[j] = 'unk'
        sentences[i] = ' '.join(words)
    return sentences, unk_c

def refine_vocab_by_subsampling(vocab, threshold_probability=0.988):
    words_to_delete = set()
    for w, c in vocab.items():
        p =  1 - (sqrt(1e-3 / c))
        if p <= threshold_probability:
            words_to_delete.add(w)
    for w in words_to_delete:
        del vocab[w]
    return vocab

def get_sentences(path, max_sentences_count=75000):
    sentences = []
    with open(path, 'r') as f:
        for text in f:
            d = json.loads(text)
            sentences.append(d["reviewText"].lower())
            if len(sentences) > (max_sentences_count):
                return sentences[:max_sentences_count]
    return sentences[:max_sentences_count]

def get_closest_words(word, word_embeddings, k=10):
    closest_words = []
    if not word in word_embeddings:
        return ['unk']
    v_1 = word_embeddings[word]
    v_1 = v_1 / np.linalg.norm(v_1)
    for nbr_word in word_embeddings.keys():
        if nbr_word in [word, 'unk', '<S>', '<E>']:
            continue
        v_2 = word_embeddings[nbr_word]
        v_2 = v_2 / np.linalg.norm(v_2)
        cosine_sim = np.dot(v_1, v_2)
        closest_words.append([cosine_sim, nbr_word])
    top_k = sorted(closest_words, reverse=True)[:10]
    return [w[1] for w in top_k]

def visualize_embeddings(word, word_embeddings):
    words_to_plot, word_vecs = [], []
    closest_words = get_closest_words(word, word_embeddings)
    for w in closest_words + [word]:
        words_to_plot.append(w)
        word_vecs.append(word_embeddings[w])
    reduced_embeddings = TSNE(n_components=2).fit_transform(word_vecs)
    # reduced_embeddings = PCA(n_components=2).fit_transform(word_vecs)
    plt.figure(figsize=(10, 10)) 
    for i in range(len(reduced_embeddings)):
        x, y = tuple(reduced_embeddings[i])
        plt.scatter(x, y)
        plt.annotate(words_to_plot[i], xy=(x, y))
    plt.title(f"Plot for closest words of {word}")
    plt.show()
    
# Negative sampling - selecting and outputting a fixed number of words sampled based on probability of occurrence
def find_neg_sampling_prob(vocab):
    denominator = sum([v ** 0.75 for v in vocab.values()])
    prob_weights = {k: (v ** 0.75) / denominator for k, v in vocab.items()}
    return prob_weights

def get_neg_samples(word, word_to_index, prob_weights, k=3):
    neg_samples = [word for _ in range(k)]
    while word in neg_samples:
        neg_samples = choices(list(word_to_index.values()), k=k, weights=list(prob_weights.values()))
    return neg_samples

# Creation of training data for model
def get_window_vectors(word_to_index, word_window, prob_weights):
    # Obtain context, word and negative sample indices to be passed as input for training
    w = len(word_window) # Note that word_window always has odd length
    word, context = word_window[w // 2], word_window[:w // 2] + word_window[ceil(w / 2):]
    # Convert sentences to list of indices
    word_index, context_vec = [word_to_index[word]], [word_to_index[c] for c in context]
    neg_samples = get_neg_samples(word, word_to_index, prob_weights, k=NEGATIVE_SAMPLES)
    return np.asarray(word_index).astype(np.float32), \
          np.asarray(context_vec).astype(np.float32), \
          np.asarray(neg_samples).astype(np.float32)

def get_data(word_to_index, prob_weights, sentences, k=5):
    d = k // 2
    word_indexes, context_vecs, all_neg_samples = [], [], []
    neg_context_sims, word_context_sims = [], []
    for i in tqdm(range(len(sentences))):
        sentence = sentences[i]
        words = tokenize_text(sentence)
        l = len(words)
        for i in range(d, l - d):
            word_index, context_vec, neg_samples = get_window_vectors(word_to_index, words[i - d:i + d + 1], prob_weights)
            word_indexes.append(word_index)
            context_vecs.append(context_vec)
            all_neg_samples.append(neg_samples)
            # Labels are 1 for the context (average vector) and 0s for each individual negative sample
            word_context_sims.append(np.asarray([1.0]).astype(np.float32))
            neg_context_sims.append(np.asarray([0] * NEGATIVE_SAMPLES).astype(np.float32))
    return np.asarray(word_indexes), np.asarray(context_vecs), np.asarray(all_neg_samples), \
            np.asarray(neg_context_sims), np.asarray(word_context_sims)
            
def save_model_and_embeddings():
    # Saving pre-trained model and embeddings
    checkpoint = ModelCheckpoint(filepath=FOLDER_PATH + 'cbow_neg_fin.hdf5', verbose=1)

    # Model creation with input, embedding and scoring layers
    cur_word = Input(shape=(1,))
    context = Input(shape=(WINDOW_SIZE-1,))
    neg_samp = Input(shape=(NEGATIVE_SAMPLES,))
    embedding_matrix_weights = np.random.uniform(low = -1 / sqrt(EMBEDDING_DIM), high = 1 / sqrt(EMBEDDING_DIM), size = (len(vocab), EMBEDDING_DIM))
    embedding_layer = Embedding(input_dim=len(vocab), output_dim=EMBEDDING_DIM, weights=[embedding_matrix_weights])

    cur_word_emb = embedding_layer(cur_word)
    context_emb = embedding_layer(context)
    neg_samp_emb = embedding_layer(neg_samp)
    cbow_vec = Lambda(lambda x: K.mean(x, axis=1), output_shape=(EMBEDDING_DIM,))(context_emb)
    word_context_sim = dot([cur_word_emb, cbow_vec], axes=-1, normalize=False)
    output1 = Dense(1, activation='sigmoid')(word_context_sim)
    neg_context_sim = dot([neg_samp_emb, cbow_vec], axes=-1, normalize=False)
    output2 = Dense(NEGATIVE_SAMPLES, activation='sigmoid')(neg_context_sim)

    model = Model(inputs=[cur_word, context, neg_samp], outputs=[output1, output2])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    print(model.summary())

    # Training
    model.fit(X, y, batch_size=256, epochs=13, callbacks=[checkpoint], verbose=2)

    word_embed_wts = embedding_layer.get_weights()[0]
    word_embeddings = {}
    for word, idx in word_to_index.items():
        word_embeddings[word] = tuple(word_embed_wts[idx].tolist())
    with open(FOLDER_PATH + 'word_embeddings_cbow_neg_fin.json', 'w') as f:
        json.dump(word_embeddings, f, indent=4)
        
    return model, word_embeddings

def load_model_and_embeddings():
    model = keras.models.load_model(FOLDER_PATH + 'cbow_neg_fin.hdf5')
    embeddings = {}
    with open(FOLDER_PATH + 'word_embeddings_cbow_neg_fin.json', 'r') as f:
        embeddings = json.load(f)

# If using google colab and drive - uncomment this
# from google.colab import drive
# drive.mount('/content/gdrive')

sentences = get_sentences(path=FOLDER_PATH + 'Electronics_5.json')

# Create vocabulary based on sentences, unk token and further sub-sampling.
vocab = get_vocabulary(sentences)
print(len(vocab))

# Uncomment below line if subsampling is needed
# vocab = refine_vocab_by_subsampling(vocab)
# print(len(vocab))

# Modify sentences using unk wherever needed
sentences, unk_c = update_sentences(vocab, sentences)
vocab['unk'] = unk_c
vocab['<S>'] = (WINDOW_SIZE - 1) * len(sentences)
vocab['<E>'] = (WINDOW_SIZE - 1) * len(sentences)
# Create word to index matrix based on vocabulary
word_to_index = {v : i for i, v in enumerate(vocab)}

# Collecting required data
prob_weights = find_neg_sampling_prob(vocab)
word_indexes, context_vecs, all_neg_samples, neg_context_sims, word_context_sims = get_data(word_to_index, prob_weights, sentences, k=WINDOW_SIZE)
X, y = [word_indexes, context_vecs, all_neg_samples], [word_context_sims, neg_context_sims]

# model, word_embeddings = save_model_and_embeddings()
# Uncomment below line and comment above line if model and embeddings are downloaded and are in folder denoted by FOLDER_PATH
model, word_embeddings = load_model_and_embeddings()

# Visualize performance of embeddings - similar words
print(get_closest_words('camera', word_embeddings))

words_list = ['good', 'camera', 'buy', 'use', 'bad', 'device']
for w in words_list:
    visualize_embeddings(w, word_embeddings)