import numpy as np
import pandas as pd


def nrmse(pred: pd.DataFrame, true: pd.DataFrame) -> float:
    return np.sqrt(
        pred.subtract(true)
            .pow(2)
            .mean(axis=0)
            .divide(true.apply(np.var, axis=0))
            .mean()
    )


def frobenius(pred: pd.DataFrame, true: pd.DataFrame):
    pass


if __name__ == '__main__':
    x = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, np.NaN]})
    y = pd.DataFrame({'a': [1.1, 2.5, 3], 'b': [2.9, np.NaN, 5.6]})
    print(nrmse(y, x))
