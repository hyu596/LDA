import json
import sys
import re
import numpy as np
from os import listdir
from os.path import isfile, join
from random import randint
import time

import lda
import lda.datasets

def counts_topic_all_words(documents, K):

    # vocab by n_topics

    counts_words_per_topics = {}

    for doc in documents.keys():
        for w in documents[doc]:
            word = re.match(r'^((.*-?)+)_[0-9]*$', w).group(1)
            if(word not in counts_words_per_topics):
                counts_words_per_topics[word] = np.zeros(K)
                counts_words_per_topics[word][documents[doc][w]] = 1
            counts_words_per_topics[word][documents[doc][w]] += 1
    return counts_words_per_topics

def counts_doc_topic(documents, K):

    # n_docs by n_topics

    counts_topics_per_docs = np.zeros((len(documents), K))
    for d in range(len(documents)):
        for w in documents[d]:
            counts_topics_per_docs[d][documents[d][w]] += 1
    return counts_topics_per_docs

def sampling_rate(documents, m, n, K, V, alpha, eta, counts_words_per_topics, counts_topics_per_docs):

    word = list(documents[m])[n]
    word_type = documents[m][word]
    true_word = re.match(r'^((.*-?)+)_[0-9]*$', word).group(1)

    rate = np.zeros(K)

    counts_topics = np.sum(counts_topics_per_docs, axis=0)
    rate_sum = 0

    is_topic_k = np.zeros(K)
    is_topic_k[word_type] = 1

    rate = (alpha + counts_topics_per_docs[m] - is_topic_k) \
             *(eta + counts_words_per_topics[true_word] - is_topic_k) \
                / (V * eta + counts_topics - is_topic_k)
    return rate / np.sum(rate)

def evaluate_beta(counts_words_per_topics, counts_topics_per_docs, valid, K, V, eta):

    beta_T = np.zeros((V, K))
    counts_topics = np.sum(counts_topics_per_docs, axis=0)

    for i, word in enumerate(valid):
        if word in counts_words_per_topics.keys():
            beta_T[i] = (eta + counts_words_per_topics[word]) \
                            / (V * eta + counts_topics)
            beta_T[i] /= np.sum(beta_T[i])
    return beta_T.T

def evaluate_M_theta(counts_topics_per_docs, alpha):

    M, K = counts_topics_per_docs.shape
    M_theta = np.zeros((M, K))
    counts_docs = np.sum(counts_topics_per_docs, axis=1)

    for m in range(M):
        theta = (alpha + counts_topics_per_docs[m]) / (K * alpha + counts_docs[m])
        M_theta[m] = theta / np.sum(theta)
    return M_theta


def most_frequent_10_words_10_topics(beta, valid):

    K, V = beta.shape
    valid_ = np.array(valid)
    for k in range(10):
        top = beta[k]
        # print(top.argsort()[-10:][::-1])
        print("Topic ", k, " :", valid_[top.argsort()[-10:][::-1]])

def calc_perplexity(X, beta, M_theta):

    loglikelihood =  - np.sum(np.multiply(X, np.log(M_theta.dot(beta))))
    return np.exp( - np.sum(np.multiply(X, np.log(M_theta.dot(beta)))) / np.sum(X)), loglikelihood

def find_topics_for_2_documents(titles, valid, M_theta, beta):

    M, K = M_theta.shape
    picked = np.random.randint(0, M, 2)
    valid_ = np.array(valid)
    print("\nClassifying topics:")
    for m in picked:
        top = np.argmax(M_theta[m])
        indices = beta[top].argsort()[-5:][::-1]
        print(titles[m], " top topic: ", top)
        print("\ttopic ", top, " : ", valid_[indices])


def main():

    X = lda.datasets.load_reuters()
    valid = list(lda.datasets.load_reuters_vocab())
    titles = list(lda.datasets.load_reuters_titles())

    # path_to_jsons = "./jsons_88_1/"
    #
    # # valid word set
    # file = open(path_to_jsons + "significant_count.result")
    # sig_count = json.load(file)
    # valid = list(sig_count.keys())

    topics = {}
    K, V = 20, len(valid)
    alpha, eta = .1, .01

    # assign topics randomly
    # files = [f for f in listdir(path_to_jsons) if isfile(join(path_to_jsons, f))]
    # i = 0
    # for file_path in files:
    #     if(file_path.find('.json') == -1):
    #         continue
    #     file = open(path_to_jsons + file_path)
    #     doc_count = json.load(file)
    #     documents = {}
    #     for word in doc_count.keys():
    #         if(word in valid):
    #             for j in range(doc_count[word]):
    #                 documents[word + str(j)] = randint(0, K-1)
    #     # print(i)
    #     topics[i] = documents
    #     i += 1

    i = 0
    for file in X:
        documents = {}
        for index, counts in enumerate(file):
            for j in range(counts):
                documents[valid[index] + "_" + str(j)] = randint(0, K-1)
        # print(i)
        topics[i] = documents
        i += 1

    # Gibbs sampling

    for it in range(500):
    # per iter

        counts_words_per_topics = counts_topic_all_words(topics, K)
        counts_topics_per_docs = counts_doc_topic(topics, K)

        start = time.time()

        # if it == 0:
        #     print("iter: ", it)
        # else:
        # print("iter: ", it, " ", np.linalg.norm(prev - temp))

        for m in range(len(topics)):
            # print("iter: ", it, " doc: ", m)
            for n in range(len(topics[m])):
                rate = sampling_rate(topics, m, n, K, V, alpha, eta, counts_words_per_topics, counts_topics_per_docs)
                topics[m][list(topics[m])[n]] = np.random.choice(K, 1, p=rate)
        end = time.time()

        if it % 10 == 0 :
            print("----------------------------")
            beta = evaluate_beta(counts_words_per_topics, counts_topics_per_docs, valid, K, V, eta)
            M_theta = evaluate_M_theta(counts_topics_per_docs, alpha)
            perplexity, loglikelihood = calc_perplexity(X, beta, M_theta)
            print("iter: ", it, " time cost: ", end - start, " perplexity: ", perplexity, " loglikelihood: ", loglikelihood)
            most_frequent_10_words_10_topics(beta, valid)

            find_topics_for_2_documents(titles, valid, M_theta, beta)
            print("----------------------------")
        else:
            print("iter: ", it, " time cost: ", end - start)


    counts_words_per_topics = counts_topic_all_words(topics, K)
    counts_topics_per_docs = counts_doc_topic(topics, K)
    beta = evaluate_beta(counts_words_per_topics, counts_topics_per_docs, valid, K, V, eta)
    most_frequent_10_words_10_topics(beta, valid)


main()
