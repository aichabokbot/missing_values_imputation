import pandas as pd
import numpy as np
from typing import Tuple


class SplitDataFrame():
    """
    Careful here, self.__data is stored as the transpose
    of the original data to ease computation.
    """

    def __init__(self, data: pd.DataFrame, na_prop: int = 0.05):
        # Drop all the columns full of NAs. We do that because some
        # time series do not start on the same day so we do not want
        # to have columns full of NAs.
        # data = data.dropna(how='all', axis=1)
        self.data = data
        self.na_prop = na_prop
        self.__data['nb_na'] = self.__data.apply(SplitDataFrame.number_of_na_in_ts, axis=0)
        self.data = self.__data.sort_values(by=['nb_na']).drop(columns=['nb_na'])
        self.train, self.valid, self.extra_na_data = self.__split_train_valid()

    @property
    def data(self) -> pd.DataFrame:
        return self.__data.T

    @data.setter
    def data(self, value):
        self.__data = value

    def __split_train_valid(self, train_proportion: int = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns a tuple in the form of :
        (
            {
                'orignal': train data
                'new': train data with new missing values
            },
            {
                'original': validation data
                'new': validation data with new missing values
            }
        )
        """
        threshold = int(train_proportion * self.__data.shape[0])
        added_na = self.generate_more_nan(self.__data).T
        train = {
            'original': self.__data.iloc[:, threshold:],
            'new': added_na.iloc[:, threshold:]
        }
        valid = {
            'original': self.__data.iloc[:, :threshold],
            'new': added_na.iloc[:, :threshold]
        }
        return train, valid, added_na

    def generate_more_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates new missing values in the data, with a proportion of `na_proportion`
        """
        n = data.shape[0] * data.shape[1]
        k = int(self.na_prop * n)
        mask = np.ones(n)
        mask[:k] = np.NaN
        np.random.shuffle(mask)
        mask = mask.reshape(data.shape)
        return data.multiply(mask)

    @staticmethod
    def number_of_na_in_ts(ts: pd.Series) -> int:
        """
        Removes all the NaNs at the beginning (assume the first value is never 
        missing), then counts the number of NaNs.
        See test below.
        """
        index_first_non_na = ts.first_valid_index()
        ts = ts[index_first_non_na:]
        return ts.isna().sum()

    @staticmethod
    def df_to_string(df: pd.DataFrame, name: str) -> str:
        """
        Displays relevant information about the DF
        """
        return f"{name} {df.shape} ({df.isna().sum().sum()} missing values) :\n" + df.head().__str__()

    def __str__(self) -> str:
        return "\n".join([
            SplitDataFrame.df_to_string(self.data, "Original data"),
            SplitDataFrame.df_to_string(self.train['original'], "Train original"),
            SplitDataFrame.df_to_string(self.train['new'], "Train new"),
            SplitDataFrame.df_to_string(self.valid['original'], "Valid original"),
            SplitDataFrame.df_to_string(self.valid['new'], "Valid new")
        ])


if __name__ == '__main__':
    # Writing quick tests here
    x = pd.Series([None, None, 1, 2, 3])
    print(SplitDataFrame.number_of_na_in_ts(x))  # 0
    x = pd.Series([None, None, 1, 2, None, 3])
    print(SplitDataFrame.number_of_na_in_ts(x))  # 1
    x = pd.Series([1, 2, 3])
    print(SplitDataFrame.number_of_na_in_ts(x))  # 0
    x = pd.Series([1, 2, None, 3])
    print(SplitDataFrame.number_of_na_in_ts(x))  # 1
    # Test for main function
    x = SplitDataFrame(pd.DataFrame(np.random.choice([1, np.NaN], size=(10, 10), p=[.95, .05])))
    print(x)
