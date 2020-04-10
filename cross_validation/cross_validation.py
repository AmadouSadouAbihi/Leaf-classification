import os, sys
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from metrics.metric import Metrics
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))


class CrossValidation:
    """
    This class does k cross validation
    """

    def __init__(self, model, hyperparameters, kfold):
        self.metrics = Metrics()
        cross_validation = StratifiedKFold(n_splits=kfold, shuffle=True)
        self.clf = GridSearchCV(model, hyperparameters, cv=cross_validation, n_jobs=-1, verbose=1)

    def fit_and_predict(self, x_train, t_train, x_test, t_test, metrics):
        prediction = self.clf.fit(x_train, t_train).best_estimator_.predict(x_test)

        if metrics == "accuracy":
            self.metrics.accuracy(self.clf, y=t_test, pred=prediction)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix(self.clf, y=t_test, pred=prediction)

        elif metrics == "roc":
            prob = self.clf.fit(x_train, t_train).best_estimator_.predict_proba(x_test)
            self.metrics.plot_roc(self.clf, y=t_test, prob=prob[:, 1])

    def get_score(self, x_test, t_test):
        return round(self.clf.score(x_test, t_test) * 100, 2)