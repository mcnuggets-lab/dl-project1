import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV



def transform(X):
    return np.power((X - TRAIN_DATA_MIN) / (TRAIN_DATA_MAX - TRAIN_DATA_MIN), 0.2)



# read in data
train_data = pd.read_csv("traindata.csv", header=None)
train_label = pd.read_csv("trainlabel.csv", header=None)
test_data = pd.read_csv("testdata.csv", header=None)

# define constants
TRAIN_DATA_MIN = train_data.min()
TRAIN_DATA_MAX = train_data.max()
EPSILON = 1e-10



# data transform
train_data = transform(train_data)
#train_data = normalize(train_data)
train_data = pd.DataFrame(train_data)
train_label = np.ravel(train_label)  # flatten the array for fitting

#summary = train_data.describe()
#print(summary)

# train-test split
train_X, test_X, train_Y, test_Y = train_test_split(train_data, train_label, test_size=0.2, 
                                   stratify=train_label, random_state=1000)


# model
clf = SVC(C=1.4, kernel="linear")
#clf = LogisticRegression(C=1.0)
clf.fit(train_X, train_Y)
train_accuracy = clf.score(train_X, train_Y)
test_accuracy = clf.score(test_X, test_Y)
print("Training accuracy: {}".format(train_accuracy))
print("Test accuracy: {}".format(test_accuracy))



# cross-validation to select parameters
parameters = {'C': [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 5, 10]}
clf_to_search = SVC(kernel="linear")
#clf_to_search = LogisticRegression()
clf2 = GridSearchCV(clf_to_search, parameters, cv=5)
clf2.fit(train_data, train_label)
print(clf2.best_estimator_)
print(clf2.best_params_)
print(clf2.best_score_)



"""
# prediction
tf_test_data = transform(test_data)
#tf_test_data = test_data
pred = clf.predict(tf_test_data)
pred = pd.Series(pred)
print(sum(pred) / pred.shape[0])
"""




































