from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier

MODELOS = {
    "RandomForest": RandomForestClassifier(n_jobs=6, random_state=42, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(n_jobs=6,random_state=42, max_iter=1500, class_weight="balanced"),
    "SVC": SVC(random_state=42, class_weight="balanced", probability=True),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_jobs=6,random_state=42, class_weight="balanced"),
    "SGDClassifier": SGDClassifier(n_jobs=6,random_state=42, class_weight="balanced"),
    "BernoulliNB": BernoulliNB(),
    "XGBoost": XGBClassifier(n_jobs=6,random_state=42)
}

PARAM_GRIDS = {
    "RandomForest": lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 10, 20, 30]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'n_jobs': 6
    },
    "DecisionTree": lambda trial: {
        'max_depth': trial.suggest_categorical('max_depth', [5, 10, 20, 30]),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4, 8]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced'])
    },
    "LogisticRegression": lambda trial: {
        'C': trial.suggest_categorical('C', [0.01, 0.1, 1.0, 10.0, 100.0]),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga', 'liblinear', 'newton-cg']),
        'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
        'max_iter': trial.suggest_categorical('max_iter', [500, 1000, 1500, 2000]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
        'n_jobs': 6
    },
    "SVC": lambda trial: {
        'C': trial.suggest_categorical('C', [0.01, 0.1, 1, 10, 100]),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 0.01, 0.1, 1]),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
        'degree': trial.suggest_categorical('degree', [2, 3, 4]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
        'probability': True
    },
    "GradientBoosting": lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 5, 10, 20]),
        'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 1.0]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    },
    "AdaBoost": lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200, 300]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.1, 0.5, 1.0, 2.0]),
        'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
    },
    "ExtraTrees": lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 10, 20, 30]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'n_jobs': 6
    },
    "SGDClassifier": lambda trial: {
        'alpha': trial.suggest_categorical('alpha', [0.0001, 0.001, 0.01, 0.1]),
        'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
        'loss': trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber', 'squared_hinge']),
        'max_iter': trial.suggest_categorical('max_iter', [1000, 2000, 3000]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
        'n_jobs': 6
    },
    "XGBoost": lambda trial: {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 5, 10, 20]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
        'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 1.0]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.7, 0.8, 1.0]),
        'gamma': trial.suggest_categorical('gamma', [0, 0.1, 0.5, 1.0]),
        'reg_alpha': trial.suggest_categorical('reg_alpha', [0, 0.1, 1]),
        'reg_lambda': trial.suggest_categorical('reg_lambda', [0.5, 1, 2]),
        'n_jobs': 6
    }
}