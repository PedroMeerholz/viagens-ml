from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from xgboost import XGBClassifier

MODELOS = {
    "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1500, class_weight="balanced"),
    "SVC": SVC(random_state=42, class_weight="balanced", probability=True),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "MLPClassifier": MLPClassifier(random_state=42, max_iter=1000),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, class_weight="balanced"),
    "LinearSVC": LinearSVC(random_state=42, max_iter=2000),
    "RidgeClassifier": RidgeClassifier(random_state=42, class_weight="balanced"),
    "SGDClassifier": SGDClassifier(random_state=42, class_weight="balanced"),
    # Novos Modelos
    "NuSVC": NuSVC(random_state=42, class_weight="balanced", probability=True),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier(random_state=42, class_weight="balanced"),
    "Perceptron": Perceptron(random_state=42, class_weight="balanced"),
    "BernoulliNB": BernoulliNB(),
    "ExtraTreeClassifier": ExtraTreeClassifier(random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(random_state=42)
}

PARAM_GRIDS = {
    "RandomForest": {
        'n_estimators': [100, 200], 
        'max_depth': [10, 20, None]
    },
    "DecisionTree": {
        'max_depth': [10, 20, 30, None], 
        'min_samples_leaf': [1, 2]
    },
    "LogisticRegression": {
        'C': [0.1, 1.0, 10.0], 
        'solver': ['lbfgs', 'saga']
    },
    "SVC": {
        'C': [0.1, 1, 10], 
        'gamma': ['scale', 'auto']
    },
    "GradientBoosting": {
        'n_estimators': [100, 200], 
        'learning_rate': [0.05, 0.1]
    },
    "AdaBoost": {
        'n_estimators': [50, 100], 
        'learning_rate': [0.1, 1.0]
    },
    "GaussianNB": {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    "MLPClassifier": {
        'hidden_layer_sizes': [(50,), (100,)], 
        'activation': ['relu', 'tanh']
    },
    "ExtraTrees": {
        'n_estimators': [100, 200], 
        'max_depth': [10, 20, None]
    },
    "LinearSVC": {
        'C': [0.1, 1, 10]
    },
    "RidgeClassifier": {
        'alpha': [0.1, 1.0, 10.0]
    },
    "SGDClassifier": {
        'alpha': [0.0001, 0.001], 
        'penalty': ['l2', 'l1']
    },
    # Grades para Novos Modelos
    "NuSVC": {
        'nu': [0.1, 0.5, 0.9], 
        'gamma': ['scale', 'auto']
    },
    "LinearDiscriminantAnalysis": {
        'solver': ['svd', 'lsqr', 'eigen']
    },
    "QuadraticDiscriminantAnalysis": {
        'reg_param': [0.0, 0.1, 0.5]
    },
    "PassiveAggressiveClassifier": {
        'C': [0.1, 1.0, 10.0]
    },
    "Perceptron": {
        'penalty': [None, 'l2', 'l1']
    },
    "BernoulliNB": {
        'alpha': [0.1, 0.5, 1.0]
    },
    "ExtraTreeClassifier": {
        'max_depth': [10, 20, None], 
        'min_samples_leaf': [1, 2]
    },
    "XGBoost": {
        'n_estimators': [100, 200], 
        'max_depth': [10, 20, None]
    }
}