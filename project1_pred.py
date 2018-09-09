import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression




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

#summary = train_X.describe()
#print(summary)



# model
clf = SVC(C=1.4, kernel="linear")
#clf = LogisticRegression(C=1)
clf.fit(train_data, train_label)
train_accuracy = (clf.predict(train_data) == train_label).sum() / train_data.shape[0]
print("Training accuracy: {}".format(train_accuracy))



# prediction
tf_test_data = transform(test_data)
#tf_test_data = test_data
pred = clf.predict(tf_test_data)
pred = pd.Series(pred)
print("Predicted +ve protion:", sum(pred) / pred.shape[0])
pred.to_csv("project1_01652721.csv", header=False, index=False)





































