import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix



def tuning_model(clf, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2, return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_train, y_train)

    return best_estimator, pd.DataFrame(grid_search.cv_results_), grid_search.best_index_, grid_search.best_score_
    

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