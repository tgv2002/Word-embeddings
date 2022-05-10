import re
import numpy as np
import json
from math import sqrt, floor, ceil
import nltk
from scipy.sparse import csr_matrix
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD, PCA
from random import sample
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import matplotlib.pyplot as plt
import pandas as pd

nltk.download('punkt')

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
    return words

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

def build_co_occurrence_matrix(sentences, word_to_index, windowSize=5):
    matrix = np.zeros((len(word_to_index), len(word_to_index)))
    for sentence in sentences:
        words = tokenize_text(sentence)
        n = len(words)
        for mid in range(windowSize, n - windowSize):
            preceding_words = words[mid - windowSize : mid]
            succeeding_words = words[mid + 1 : mid + windowSize + 1]
            mid_word = words[mid]
            if word_to_index.get(mid_word) is None:
                continue
            for w in preceding_words + succeeding_words:
                if word_to_index.get(w) is None:
                    continue
                i, j = word_to_index[mid_word], word_to_index[w]
                matrix[i, j] += 1
    return matrix

def get_word_embeddings(M, word_to_index):
    u = TruncatedSVD(n_components=200).fit_transform(csr_matrix(M))
    print(u.shape)
    embeddings = {w: tuple(u[idx]) for w, idx in word_to_index.items()}
    with open(FOLDER_PATH + 'word_embeddings_svd.json', 'w') as f:
    # with open('./word_embeddings_svd.json', 'w') as f:
        json.dump(embeddings, f, indent=4)
    return embeddings

def get_sentences(path, max_sentences_count=130000):
    sentences = []
    with open(path, 'r') as f:
        for text in f:
            d = json.loads(text)
            sentences.append(d["reviewText"].lower())
            if len(sentences) > (2*max_sentences_count):
                return sample(sentences, k=max_sentences_count)
    return sample(sentences, k=max_sentences_count)

def get_closest_words(word, word_embeddings, k=10):
    closest_words = []
    if not word in word_embeddings:
        return ['unk']
    v_1 = word_embeddings[word]
    v_1 = v_1 / np.linalg.norm(v_1)
    for nbr_word in word_embeddings.keys():
        if nbr_word in [word, 'unk']:
            continue
        v_2 = word_embeddings[nbr_word]
        v_2 = v_2 / np.linalg.norm(v_2)
        cosine_sim = np.dot(v_1, v_2)
        closest_words.append([cosine_sim, nbr_word])
    top_k = sorted(closest_words, reverse=True)[:10]
    return [w[1] for w in top_k]

# Comparison with pre-trained embeddings of existing models
def pretrained_model_words(word):
    existing_model = KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)
    similar_words = existing_model.most_similar(word)
    return [s[0] for s in similar_words]
    
def load_embeddings():
    with open(FOLDER_PATH + 'word_embeddings_svd.json', 'r') as f:
        word_embeddings = json.load(f)
    return word_embeddings
        
def visualize_embeddings(word, word_embeddings):
    words_to_plot, word_vecs = [], []
    closest_words = get_closest_words(word, word_embeddings)
    for w in closest_words + [word]:
        words_to_plot.append(w)
        word_vecs.append(word_embeddings[w])
    reduced_embeddings = TSNE(n_components=2, random_state=123).fit_transform(word_vecs)
    # reduced_embeddings = PCA(n_components=2, random_state=123).fit_transform(word_vecs)
    plt.figure(figsize=(10, 10)) 
    for i in range(len(reduced_embeddings)):
        x, y = tuple(reduced_embeddings[i])
        plt.scatter(x, y)
        plt.annotate(words_to_plot[i], xy=(x, y))
    plt.title(f"Plot for closest words of {word}")
    plt.show()   

# If using google colab and drive - uncomment this
# from google.colab import drive
# drive.mount('/content/gdrive')

sentences = get_sentences(path=FOLDER_PATH + 'Electronics_5.json')

vocab = get_vocabulary(sentences)
# vocab = refine_vocab_by_subsampling(vocab)
# print(len(vocab))
sentences, unk_c = update_sentences(vocab, sentences)
vocab['unk'] = unk_c
word_to_index = {v : i for i, v in enumerate(vocab)}

# M = build_co_occurrence_matrix(sentences, word_to_index)
# word_embeddings = get_word_embeddings(M, word_to_index)
# with open(FOLDER_PATH + 'word_embeddings_svd.json', 'w') as f:
#     json.dump(word_embeddings, f)

# Comment above 4 lines and uncomment below line if embeddings are already calculated and downloaded and in correct folder
word_embeddings = load_embeddings()

print(get_closest_words('camera', word_embeddings))

words_list = ['good', 'camera', 'buy', 'use', 'bad', 'device']
for w in words_list:
    visualize_embeddings(w, word_embeddings)

print(pretrained_model_words('camera'))