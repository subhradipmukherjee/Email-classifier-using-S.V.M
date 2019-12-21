import os
import io
import collections,numpy
import pandas as pd
import array as arr
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('C:/Users/uttam3in/Desktop/DataScience-Python3/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('C:/Users/uttam3in/Desktop/DataScience-Python3/emails/ham', 'ham'))

from sklearn import svm, datasets
C = 1.0
train, test = train_test_split(data, test_size=0.2)
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(train['message'].values)
targets = train['class'].values
svc = svm.SVC(kernel='linear', C=C).fit(counts, targets)
test_counts = vectorizer.transform(test['message'].values)
pred = svc.predict(test_counts)
print(pred)
print(accuracy_score(test['class'].values,pred))