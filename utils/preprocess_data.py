from dataclasses import field
import warnings
from sklearn.preprocessing import StandardScaler
from utils.feature_extraction import CapacityAndSOCCalculation, OCVDVACalculation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.data_import import import_datafile
from typing import Dict, List, Optional
import numpy as np
import os
import zipfile
import re
import pandas as pd


class PreprocessData:
    def __init__(self, data_params):
        self.data_params = data_params
        self._relevant_csv_list = self._provide_relevant_filenames(data_params)

    def load_dfs(self, data_path):

        relevant_filenames = self._relevant_csv_list
        df_list = []
        filenames = []
        all_zips = os.listdir(data_path)
        for zip_file in all_zips:
            zip_contents = zipfile.ZipFile(os.path.join(data_path, zip_file))
            for file_path in zip_contents.namelist():
                if file_path.endswith('.csv'):
                    filename = file_path.split("/")[-1]
                    if filename in relevant_filenames:
                        with zip_contents.open(file_path) as csv_file:
                            df = import_datafile(csv_file)
                            aging_type, testpoint, rpt, temperature = self.extract_parameters(filename)
                            df = df.assign(
                                aging_type=aging_type,
                                testpoint=int(testpoint),
                                rpt=rpt,
                                temperature=int(temperature)
                            )
                            df = self._identify_test_part(df)
                            if self.data_params.time_res:
                                df = self.interpolate_data(df, time_res=self.data_params.time_res)

                            if self.data_params.add_feature_cols:
                                capacity_soc_cal = CapacityAndSOCCalculation()
                                df = capacity_soc_cal.add_capacity_soc_cols(df)
                                ocv_dva_calc = OCVDVACalculation()
                                df = ocv_dva_calc.add_ocv_dva(df)

                            filenames.append(filename)
                            df_list.append(df)

        print(f"The number of files loaded: {len(df_list)}",
              f"File Names are: {filenames}")

        df = pd.concat(df_list, axis=0)
        return df

    def _provide_relevant_filenames(self, data_params):

        stages = data_params.stages
        aging_types = data_params.aging_types
        testpoints = data_params.testpoints
        cells = data_params.cells
        rpts = data_params.reference_performance_test
        temps = data_params.temps
        file_numbers = np.arange(0, 20, 1)

        relevant_csv_list = []

        for aging_type in aging_types:
            for testpoint in testpoints:
                for cell in cells:
                    for file_number in file_numbers:
                        for rpt in rpts:
                            for temp in temps:
                                relevant_csv_list.append(f"TP_{aging_type}{testpoint:02d}_{cell:02d}_{file_number:02d}_{rpt}_T{temp}.csv")
        self._relevant_csv_list = relevant_csv_list
        return relevant_csv_list

    def extract_parameters(self, file_name):

        file_name = file_name.split('/')[-1]
        file_name_list = file_name.split("_")

        aging_type =re.sub(r'[^a-zA-Z]', '', file_name_list[1])
        testpoint = re.sub(r'[^0-9]', '', file_name_list[1])
        rpt = file_name_list[-2]
        temperature = re.sub(r'[^0-9]', '', file_name_list[-1])

        return aging_type, testpoint, rpt, temperature

    def _identify_test_part(self, df):

        capacity_index = df[(df['step_type'] == 21)].index[0]
        pocv_starting_index = df[df['step_type'] > 30].index[0]
        hppc_starting_index = df[df['step_type'] > 100].index[0]

        conditions = [
            (df.index < capacity_index),
            (df.index >= capacity_index) & (df.index <= pocv_starting_index) ,
            (df.index >= pocv_starting_index) & (df.index <= hppc_starting_index),
            (df.index > hppc_starting_index)
        ]

        testparts = ['warming_up', 'capacity_cycle', 'pocv', 'hppc']

        df['testpart'] = np.select(conditions, testparts, default="NA")

        return df

    def interpolate_data(self, df,
                         time_col: str = "run_time",
                         time_res: float = 1):

        if any(df.duplicated(subset=[time_col])):
            warnings.warn(f"Caution: The column {time_col} contains {len(df) - len(df[~(df.duplicated(subset=[time_col]))])} duplicate values."
                  f"\nProceeding by removing the duplicates.", UserWarning)
            df = df[~(df.duplicated(subset=[time_col]))]

        df = df.sort_values(time_col)

        constant_time_series = np.arange(int(df.index.min()), int(df.index.max())+1, time_res)

        numeric_cols = df.select_dtypes(include="number").columns
        non_numeric_cols = df.select_dtypes(exclude="number").columns

        interpolated_df = pd.DataFrame({time_col: constant_time_series})
        for col in numeric_cols:
            new_col_data = np.interp(constant_time_series, df[time_col], df[col])
            interpolated_df[col] = new_col_data

        interpolated_df = pd.merge_asof(interpolated_df, df[(non_numeric_cols).tolist() + [time_col]], on=time_col, direction='backward')

        return interpolated_df

    def plot(self, df,
             x_axis='run_time',
             y_axes:list = None,
             conditions: dict=None,
             downsampling_factor: int = 100):

        if y_axes is None:
            y_axes = ['c_cur', 'c_vol']
        if conditions:
            for col, value in conditions.items():
                df = df[df[col] == value]

        for y_axis in y_axes:
            plt.plot(df[x_axis][::downsampling_factor], df[y_axis][::downsampling_factor], label=y_axis)
        plt.xlabel(str(x_axis))
        plt.legend()
        plt.show()

    def add_sequence_data(self, df: pd.DataFrame,
                          seq_cols: Optional[List[str]] = None,
                          num_points: int = 5) -> pd.DataFrame:

        if not seq_cols:
            seq_cols = ['c_cur', 'c_vol']

        shifted_cols = {}
        for seq_col in seq_cols:
            for num_point in range(1, num_points + 1):
                shifted_cols[f"{seq_col}-{num_point}"] = df[seq_col].shift(num_point)
        df = pd.concat([df, pd.DataFrame(shifted_cols, index=df.index)], axis=1).fillna(0)
        return df

    def standardize_data(self, df, feature_cols,
                         scaler=None):
        if not scaler:
            scaler = StandardScaler()
            scaler.fit(df[feature_cols])

        df[feature_cols] = scaler.transform(df[feature_cols])
        return df, scaler

    def fitler_data(self, df: pd.DataFrame, filtering_conditions: Dict):

        for col, value in filtering_conditions.items():
            df = df[df[col] == value]
        return df

