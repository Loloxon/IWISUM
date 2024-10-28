import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    median = np.median(train_data)
    std = np.std(train_data)
    res = []
    print(f"Median: {median}; std: {std}")
    for d in test_data:
        if median-3*std<d<median+3*std:
            res.append(0)
        else:
            res.append(1)
    return res
