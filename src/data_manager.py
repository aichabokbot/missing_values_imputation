import pandas as pd
import pathlib
import os
from typing import Tuple
from src.split_data_frame import SplitDataFrame
from src.metrics import nrmse


class DataManager:
    DATA_PATH = '../data'
    TIME_SERIES_PATH = 'data_set_challenge.csv'
    TYPE_PATH = 'final_mapping_candidat.csv'

    def __init__(self, na_prop: int = 0.05):
        self.na_prop = na_prop
        here = pathlib.Path(__file__).parent.absolute()
        time_series_data = pd.read_csv(
            os.path.join(
                here,
                DataManager.DATA_PATH,
                DataManager.TIME_SERIES_PATH
            ),
            index_col=0
        )
        self.original_data = time_series_data
        self.original_data.columns = [
            int(i) for i in list(self.original_data.columns)
        ]
        self.original_data.index = pd.to_datetime(self.original_data.index)
        self.original_data = self.original_data.sort_index()
        type_data = pd.read_csv(os.path.join(
            here,
            DataManager.DATA_PATH,
            DataManager.TYPE_PATH
        )).drop(columns=['Unnamed: 0'])
        time_series_data = time_series_data.T
        time_series_data.index = time_series_data.index.astype('int64')
        full_data = time_series_data.merge(
            right=type_data,
            left_index=True,
            right_on='mapping_id'
        )
        self.split_data = {
            k: SplitDataFrame(
                df.drop(columns=['Type', 'mapping_id']),
                na_prop=self.na_prop
            )
            for k, df in full_data.groupby('Type')
        }
        self.types = type_data["Type"].unique()

    def last_value_carried_forward(self) -> Tuple[pd.DataFrame, float]:
        """
        Implementation of the Last Value Carried forward method as
        a dummy example.
        """
        pred_dfs = []
        for k, v in self.split_data.items():
            pred_split = v.extra_na_data.fillna(method='ffill')
            pred_dfs.append(pred_split)
        pred = pd.concat(pred_dfs, axis=1)
        return pred, nrmse(pred, self.original_data)

    def __getitem__(self, type: str) -> SplitDataFrame:
        return self.split_data.get(type, None)


if __name__ == '__main__':
    dm = DataManager(na_prop=0.05)
    pred, nrmse_value = dm.last_value_carried_forward()
    print(nrmse_value)
