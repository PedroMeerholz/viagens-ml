import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
  

def predict(clf, X_test, y_test):
    prediction = clf.predict(X_test)
    acc_test = accuracy_score(y_test, prediction)
    precision_test = precision_score(y_test, prediction, average='weighted')
    recall_test = recall_score(y_test, prediction, average='weighted')
    f1_test = f1_score(y_test, prediction, average='weighted')

    return prediction, acc_test, precision_test, recall_test, f1_test


def get_classification_report(y_test, prediction):
    report = classification_report(y_test, prediction, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.reset_index(inplace=True)
    report.columns = ['reference', 'precision', 'recall', 'f1-score', 'support']
    return report


def get_confusion_matrix(y, prediction):
    return confusion_matrix(y, prediction)