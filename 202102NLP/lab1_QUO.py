import nltk
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from sklearn.svm import SVC
import re
from sklearn.metrics import accuracy_score
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd

class EmotionClassification:
    # global variable
    train_label = None

    tf_idf = None

    sims = None

    dictionary = None

    stop_words = set(stopwords.words('english'))

    # for every input sentence,
    # lemmatized and removed stop_words
    # finally, return a words_list
    def preprocess_sent(self, sent):
        sent = [w.lower() for w in word_tokenize(sent)]
        # Stemming and Lemmatization
        # porter = nltk.PorterStemmer()
        # sent = [porter.stem(t) for t in sent]
        wnl = nltk.WordNetLemmatizer()
        sent = [wnl.lemmatize(t) for t in sent]
        sent = [t for t in sent if t not in self.stop_words]
        return sent

    # read the file with the special format(;)
    # return the words_list and label_list(if any)
    def read_file(self, address):
        gen_docs = []
        label_list = []
        with open(address) as f:
            docs = f.readlines()
            for doc in docs:
                # because every line is a sentence. So, the first thing is to tokenization
                # and remove stop word byntlk.stopword
                doc = doc.replace("\n", "")
                doc = doc.split(";")
                doc[0] = self.preprocess_sent(doc[0])
                gen_docs.append(doc[0])
                if (len(doc) == 1):
                    label_list=None
                    continue
                label_list.append(doc[1])
            return np.array(gen_docs), np.array(label_list)

    # for an input sentence
    # Convert `document` into the bag-of-words ang then convert it to tf-idf vector
    # calculate the cos similarity between the training data
    # count k maximum's label
    # and return the majority of k's label(like k-nn)
    def classify_one_sentence(self, sent, k_num=8):
        query_doc_bow = self.dictionary.doc2bow(sent)
        # perform a similarity query against the corpus
        query_doc_tf_idf = self.tf_idf[query_doc_bow]
        # print(document_number, document_similarity)
        temp = np.array(self.sims[query_doc_tf_idf])
        max_index_list = np.argpartition(temp, (-1) * k_num)[k_num * (-1):]
        temp = self.train_label[max_index_list]
        word_counts = Counter(temp)
        # 出现频率最高的1个单词
        top_one = word_counts.most_common(1)
        result = top_one[0][0]
        return result

    #  for an input file
    #  Convert `document` into the bag-of-words ang then convert it to tf-idf vector
    def fit(self, train_address='./data/train.txt'):
        print("Training...")
        train_docs, self.train_label = self.read_file(train_address)
        self.dictionary = gensim.corpora.Dictionary(train_docs)
        # print(dictionary.token2id)
        corpus = [self.dictionary.doc2bow(train_doc) for train_doc in train_docs]
        self.tf_idf = gensim.models.TfidfModel(corpus)
        # for doc in tf_idf[corpus]:
        #     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
        self.sims = gensim.similarities.Similarity('result/', self.tf_idf[corpus], num_features=len(self.dictionary))
        print('')

    # use a loop to classify_one_sentence
    # and generate the acc between the real label and the predicted value
    def predict_file(self, target_address='./data/val.txt'):

        feature_test, label_test = self.read_file(target_address)

        label_prd = []
        for doc in feature_test:
            result = self.classify_one_sentence(doc, k_num=13)
            label_prd.append(result)
        print(accuracy_score(label_test, label_prd))
        return label_prd
    # similar as the function predict_file
    # but aim to save the result as required format
    def print_prd_file(self, target_address='./data/test_prediction.txt.txt'):
        feature_test, label_test = self.read_file(target_address)

        label_prd = []
        for doc in feature_test:
            result = self.classify_one_sentence(doc, k_num=13)
            label_prd.append([result])
        data = pd.DataFrame(label_prd)
        with open('./result/result.csv', 'wb') as f:
            data.to_csv(f,index=False,header=False)


    def example(self):
        self.fit()
        self.predict_file()

# sample
EmotionClassification().example()
