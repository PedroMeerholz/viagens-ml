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
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    "DecisionTree": {
        'max_depth': [5, 10, 20, 30, None],
        'min_samples_leaf': [1, 2, 4, 8],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'class_weight': ['balanced', None]
    },
    "LogisticRegression": {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['lbfgs', 'saga', 'liblinear', 'newton-cg'],
        'penalty': ['l2', 'l1', 'elasticnet', 'none'],
        'max_iter': [500, 1000, 1500, 2000],
        'class_weight': ['balanced', None]
    },
    "SVC": {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],
        'class_weight': ['balanced', None],
        'probability': [True]
    },
    "GradientBoosting": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 10, 20, None],
        'subsample': [0.7, 0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    "GaussianNB": {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },
    "MLPClassifier": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [500, 1000, 2000]
    },
    "ExtraTrees": {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    "LinearSVC": {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'loss': ['squared_hinge'],
        'max_iter': [1000, 2000, 3000],
        'class_weight': ['balanced', None]
    },
    "RidgeClassifier": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
        'class_weight': ['balanced', None]
    },
    "SGDClassifier": {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
        'max_iter': [1000, 2000, 3000],
        'class_weight': ['balanced', None]
    },
    # Grades para Novos Modelos
    "NuSVC": {
        'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],
        'class_weight': ['balanced', None],
        'probability': [True]
    },
    "LinearDiscriminantAnalysis": {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 1.0]
    },
    "QuadraticDiscriminantAnalysis": {
        'reg_param': [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        'store_covariance': [True, False],
        'tol': [1e-4, 1e-3, 1e-2]
    },
    "PassiveAggressiveClassifier": {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'loss': ['hinge', 'squared_hinge'],
        'max_iter': [1000, 2000, 3000],
        'class_weight': ['balanced', None]
    },
    "Perceptron": {
        'penalty': [None, 'l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000, 3000],
        'class_weight': ['balanced', None]
    },
    "BernoulliNB": {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
        'fit_prior': [True, False]
    },
    "ExtraTreeClassifier": {
        'max_depth': [5, 10, 20, 30, None],
        'min_samples_leaf': [1, 2, 4, 8],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'class_weight': ['balanced', None]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 10, 20, None],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0.5, 1, 2]
    }
}