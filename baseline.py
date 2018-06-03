import numpy as np
import lda
import lda.datasets
import time

from pre_proc import read_file
from scipy.sparse import *

def input_fn(filename=None):
    if filename == None:
        X = lda.datasets.load_reuters()
        valid = list(lda.datasets.load_reuters_vocab())
        titles = list(lda.datasets.load_reuters_titles())
    return X, valid

def input_fn_ny():

    '''read data'''
    # contains index of words appearing in that document and the number of times they appear
    with open('data/nyt_data.txt') as f:
        documents = f.readlines()
    documents = [x.strip().strip('\n').strip("'") for x in documents]

    # contains vocabs with rows as index
    with open('data/nyt_vocab.dat') as f:
        vocabs = f.readlines()
    vocabs = [x.strip().strip('\n').strip("'") for x in vocabs]

    '''create matrix X'''
    numDoc = 8447
    numWord = 3012
    X = np.zeros([numWord,numDoc])

    for col in range(len(documents)):
        for row in documents[col].split(','):
            X[int(row.split(':')[0])-1,col] = int(row.split(':')[1])
    X = X.T.astype(int)
    return X, vocabs


#start_time = time.time()
#X = read_file('./data/docword.nytimes.txt')
X, vocabs = input_fn_ny()
#np.save('./data/nytimes', X)
#print("end")

#print("Time to pre_proc: ", time.time() - start_time)
#print(X.shape)

# X = lda.datasets.load_reuters()

model = lda.LDA(n_topics=20, n_iter=1000, random_state=1)

# X = X.T.astype(int)

start_time = time.time()
model.fit(X)  # model.fit_transform(X) is also available
print("Time: ", time.time() - start_time)


# topic_word = model.topic_word_  # model.components_ also works
# print(topic_word)
# print(topic_word.shape)
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word[:20]):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# doc_topic = model.doc_topic_
# for i in range(10):
#     print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
