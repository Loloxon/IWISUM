from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ex2.solution2 import detect as detect2
from ex3.solution3 import detect as detect3
from utils import binary2neg_boolean
import numpy as np

SEED = 1


def detect_cov(data: np.ndarray, outliers_fraction: float) -> list:
    cov = EllipticEnvelope(random_state=0).fit(data)
    res_neg = cov.predict(data)
    res = []
    for d in res_neg:
        if d==-1:
            res.append(1)
        else:
            res.append(0)
    return res

def detect_ocsvm(data: np.ndarray, outliers_fraction: float) -> list:
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf")
    clf.fit(data)
    y_pred_train = clf.predict(data)
    return binary2neg_boolean(y_pred_train)


def detect_iforest(data: np.ndarray, outliers_fraction: float) -> list:
    clf = IsolationForest(random_state=0, contamination=outliers_fraction).fit(data)
    res_neg = clf.predict(data)
    res = []
    for d in res_neg:
        if d==-1:
            res.append(1)
        else:
            res.append(0)
    return res


def detect_lof(data: np.ndarray, outliers_fraction: float) -> list:
    clf = LocalOutlierFactor(n_neighbors=400, contamination=outliers_fraction)
    res_neg = clf.fit_predict(data)
    res = []
    for d in res_neg:
        if d==-1:
            res.append(1)
        else:
            res.append(0)
    return res

