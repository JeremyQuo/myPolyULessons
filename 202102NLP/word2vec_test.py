import nltk
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import model_selection, svm
import pandas as pd
import numpy as np
from gensim.models import word2vec
from sklearn.metrics import accuracy_score


np.random.seed(500) # keep the consistency of the results

def SVMClassifier(Train_X, Test_X, Train_Y, Test_Y):
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    Test_X=np.nan_to_num(Test_X)
    SVM.fit(Train_X,Train_Y)
    prediction = SVM.predict(Test_X)
    accuracy = accuracy_score(prediction, Test_Y)

    print('SVM Accuracy Score -> {}%'.format(accuracy*100))


def PreProceccing(data, labels):
    # change the labels to numbers
    # change all the text to lower case
    data = [sent.lower() for sent in data]
    # broke sentences into sets of words
    data = [word_tokenize(sent) for sent in data]
    # remove stop words, do lemmatization
    clean_data = []
    word_Lemmatized = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    for idx, sent in enumerate(data):
        new_sent = [word_Lemmatized.lemmatize(w) for w in sent if w not in stop_words]
        clean_data.append(new_sent)

    assert len(clean_data) == len(labels)

    # chang the text to numerical values
    # sklearn.feature_extraction.text.TfidfVectorier
    # gensim.models.word2vec
    WordToVec = word2vec.Word2Vec(clean_data, size=20, window=20, min_count=2, workers=4)

    vocab = WordToVec.wv.vocab.keys()
    final_data = np.empty([0,20], dtype = float)

    # I love cs:
    # [0.3, 0.2, -0.7] --> I
    # [0.4, -0.2, 0.7] --> love
    # [0.3, 0.4, -0.9] --> cs
    # [(1/3), (0.4/3), (-0.9)/3]

    for sent in clean_data:
        sent = [WordToVec.wv[w] for w in sent if w in vocab]
        sent = np.mean(sent, axis=0) # sum, max,
        final_data=np.insert(final_data, 0, values=sent, axis=0)
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(final_data, labels,
                                                                test_size=0.2)

    return Train_X, Test_X, Train_Y, Test_Y


def read_data():
    data = pd.read_csv('data/train.txt', header=None,sep=';')
    data = shuffle(data)
    data.columns=['text','tag']
    texts = data['text'].tolist()
    labels = data['tag'].tolist()

    assert len(texts) == len(labels)

    return texts, labels

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    texts, labels = read_data()
    Train_X, Test_X, Train_Y, Test_Y = PreProceccing(texts, labels)
    SVMClassifier(Train_X, Test_X, Train_Y, Test_Y)

