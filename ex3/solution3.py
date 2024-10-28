from sklearn import svm
from utils import binary2neg_boolean
import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    clf = svm.OneClassSVM(nu=0.002, kernel="rbf")
    clf.fit(train_data)
    y_pred_train = clf.predict(test_data)
    return binary2neg_boolean(y_pred_train)

