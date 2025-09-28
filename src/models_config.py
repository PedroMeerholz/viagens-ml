from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

MODELOS = {
    "RandomForest": {'default_params': {'random_state': 42, 'class_weight': 'balanced'}},
    "DecisionTree": {'default_params': {'random_state': 42, 'class_weight': 'balanced'}},
    "LogisticRegression": {'default_params': {'random_state': 42, 'max_iter': 1500, 'class_weight': 'balanced'}},
    "KNN": {'default_params': {}},
    "SVC": {'default_params': {'random_state': 42, 'class_weight': 'balanced', 'probability': True}},
    "GradientBoosting": {'default_params': {'random_state': 42}},
    "AdaBoost": {'default_params': {'random_state': 42}},
    "GaussianNB": {'default_params': {}},
    "MLPClassifier": {'default_params': {'random_state': 42, 'max_iter': 1000}},
    "ExtraTrees": {'default_params': {'random_state': 42, 'class_weight': 'balanced'}},
    "LinearSVC": {'default_params': {'random_state': 42, 'max_iter': 2000}},
    "RidgeClassifier": {'default_params': {'random_state': 42, 'class_weight': 'balanced'}},
    "SGDClassifier": {'default_params': {'random_state': 42, 'class_weight': 'balanced'}},
    # Novos Modelos
    "NuSVC": {'default_params': {'random_state': 42, 'class_weight': 'balanced', 'probability': True}},
    "LinearDiscriminantAnalysis": {'default_params': {}},
    "QuadraticDiscriminantAnalysis": {'default_params': {}},
    "PassiveAggressiveClassifier": {'default_params': {'random_state': 42, 'class_weight': 'balanced'}},
    "Perceptron": {'default_params': {'random_state': 42, 'class_weight': 'balanced'}},
    "BernoulliNB": {'default_params': {}},
    "NearestCentroid": {'default_params': {}},
    "ExtraTreeClassifier": {'default_params': {'random_state': 42, 'class_weight': 'balanced'}},
}

PARAM_GRIDS = {
    "RandomForest": {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    "DecisionTree": {'max_depth': [10, 20, 30, None], 'min_samples_leaf': [1, 2]},
    "LogisticRegression": {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'saga']},
    "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    "SVC": {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    "GradientBoosting": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
    "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]},
    "GaussianNB": {'var_smoothing': [1e-9, 1e-8, 1e-7]},
    "MLPClassifier": {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']},
    "ExtraTrees": {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    "LinearSVC": {'C': [0.1, 1, 10]},
    "RidgeClassifier": {'alpha': [0.1, 1.0, 10.0]},
    "SGDClassifier": {'alpha': [0.0001, 0.001], 'penalty': ['l2', 'l1']},
    # Grades para Novos Modelos
    "NuSVC": {'nu': [0.1, 0.5, 0.9], 'gamma': ['scale', 'auto']},
    "LinearDiscriminantAnalysis": {'solver': ['svd', 'lsqr', 'eigen']},
    "QuadraticDiscriminantAnalysis": {'reg_param': [0.0, 0.1, 0.5]},
    "PassiveAggressiveClassifier": {'C': [0.1, 1.0, 10.0]},
    "Perceptron": {'penalty': [None, 'l2', 'l1']},
    "BernoulliNB": {'alpha': [0.1, 0.5, 1.0]},
    "NearestCentroid": {'shrink_threshold': [None, 0.1, 0.5]},
    "ExtraTreeClassifier": {'max_depth': [10, 20, None], 'min_samples_leaf': [1, 2]},
}