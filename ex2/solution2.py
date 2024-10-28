import numpy as np
from sklearn.covariance import MinCovDet


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    cov = MinCovDet(random_state=0).fit(train_data)

    max_dist = max(cov.mahalanobis(train_data))
    test_dist = cov.mahalanobis(test_data)
    res = []
    for d in test_dist:
        if d<max_dist:
            res.append(0)
        else:
            res.append(1)
    return res