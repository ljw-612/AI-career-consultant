import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import joblib


def load_data():
    svm_data = pd.read_csv('../data/svm_data.csv')
    # sample 15000 data
    svm_data = svm_data.sample(15000)
    X_train, X_test, y_train, y_test = train_test_split(svm_data.drop('Tag', axis=1), svm_data['Tag'], test_size=0.3)
    return X_train, X_test, y_train, y_test

def model(X_train, X_test, y_train, y_test):
    # create a SVM model
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return clf, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm):


    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

def main():
    X_train, X_test, y_train, y_test = load_data()
    clf, cr, cm = model(X_train, X_test, y_train, y_test)
    print(cr)
    print(cm)
    plot_confusion_matrix(cm)
    # save the model
    joblib.dump(clf, '../model/svm_model.pkl')


if __name__ == '__main__':
    main()