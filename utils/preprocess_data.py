import warnings
from utils.data_import import import_datafile
from typing import Dict, List, Optional
import numpy as np
import os
import zipfile
import re
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from utils.feature_extraction import MultiStageDataCapacityAndSOCCalculation, MultiStageDataOCVDVACalculation, CalceCapacityAndSOCCalculation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class BasePreProcess:
    def __init__(self):
        pass

    def interpolate_data(self, df,
                         time_col: str,
                         time_res: float = 1):

        if any(df.duplicated(subset=[time_col])):
            warnings.warn(
                f"Caution: The column {time_col} contains {len(df) - len(df[~(df.duplicated(subset=[time_col]))])} duplicate values."
                f"\nProceeding by removing the duplicates.", UserWarning)
            df = df[~(df.duplicated(subset=[time_col]))]

        df = df.sort_values(time_col)

        constant_time_series = np.arange(int(df[time_col].min()), int(df[time_col].max()) + 1, time_res)

        numeric_cols = df.select_dtypes(include="number").columns
        non_numeric_cols = df.select_dtypes(exclude="number").columns

        interpolated_df = pd.DataFrame({time_col: constant_time_series})
        for col in numeric_cols:
            new_col_data = np.interp(constant_time_series, df[time_col], df[col])
            interpolated_df[col] = new_col_data

        interpolated_df = pd.merge_asof(interpolated_df, df[(non_numeric_cols).tolist() + [time_col]], on=time_col,
                                        direction='backward')

        return interpolated_df

    def standardize_data(self, df, feature_cols,
                         scaler=None):
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(df[feature_cols])

        df[feature_cols] = scaler.transform(df[feature_cols])
        return df, scaler

    def filter(self, df: pd.DataFrame, filtering_conditions: Dict):

        for col, value in filtering_conditions.items():
            df = df[df[col].isin(value)].copy()
        return df

    def plot(self, df,
             x_axis: str,
             y_axes: list,
             conditions: dict=None,
             downsampling_factor: int = 100):

        if conditions:
            for col, value in conditions.items():
                df = df[df[col] == value]

        fig, ax = plt.subplots(len(y_axes), 1, sharex=True)
        if not isinstance(ax, (list, np.ndarray)):
            ax = [ax]

        for i, y_axis in enumerate(y_axes):
            ax[i].plot(df[x_axis][::downsampling_factor], df[y_axis][::downsampling_factor], 'o', label=y_axis)
            ax[i].set_ylabel(str(y_axis))
            ax[i].grid(True)
        plt.xlabel(str(x_axis))
        plt.legend()
        plt.show()

    def add_sequence_data(self, df: pd.DataFrame,
                          seq_cols: List[str],
                          history_length: int = 5) -> pd.DataFrame:

        shifted_cols = {}
        for seq_col in seq_cols:
            for num_point in range(1, history_length + 1):
                shifted_cols[f"{seq_col}-{num_point}"] = df[seq_col].shift(num_point)
        df = pd.concat([df, pd.DataFrame(shifted_cols, index=df.index)], axis=1).dropna(axis=0)
        return df

    def add_sequence_data_per_col(self, df: pd.DataFrame,
                          seq_cols: List[str],
                          history_length: List[int]) -> pd.DataFrame:

        shifted_cols = {}
        for i, seq_col in enumerate(seq_cols):
            for num_point in range(1, history_length[i] + 1):
                shifted_cols[f"{seq_col}-{num_point}"] = df[seq_col].shift(num_point)
        df = pd.concat([df, pd.DataFrame(shifted_cols, index=df.index)], axis=1).dropna(axis=0)
        return df


class PreprocessMultiStageData(BasePreProcess):
    def __init__(self, data_params):
        super().__init__()

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
                            aging_type, testpoint, rpt, temperature = self._extract_parameters(filename)
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
                                capacity_soc_cal = MultiStageDataCapacityAndSOCCalculation()
                                df = capacity_soc_cal.add_capacity_soc_cols(df)
                                ocv_dva_calc = MultiStageDataOCVDVACalculation()
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

    def _extract_parameters(self, file_name):

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


class PreprocessCalceA123(BasePreProcess):
    def __init__(self, calce_data_params):
        super().__init__()
        self.calce_data_params = calce_data_params

    def load_dfs(self, data_path):

        df_list = []
        for file_path in glob.glob(os.path.join(data_path, "**", "*.xlsx"), recursive=True):
            file_name = os.path.basename(file_path)
            df = pd.read_excel(file_path, sheet_name="Sheet1")
            df = self.interpolate_data(df, time_col="Test_Time(s)", time_res=self.calce_data_params.time_res)
            temp = self._extract_parameters(file_name)
            df['amb_temp'] = temp
            df = self._identify_test_part(df)
            df = self.soc_calculation(df)
            df_list.append(df)

        df = pd.concat(df_list, axis=0)
        return df

    def _extract_parameters(self, file_name):
        temperature = file_name.split('FUDS-')[-1].split("-")[0]
        return int(temperature)

    def _identify_test_part(self, df):

        charging_start_indices, charging_end_indices = self._identify_constant_charging_indices(df)
        charging_indices = [idx for i in range(len(charging_start_indices)) for idx in (charging_start_indices[i], charging_end_indices[i])] + [df.index[-1]]

        df['testpart'] = "None"
        testparts = ["charging_1", "DST", "charging_2", "US06", "charging_3", "FUD"]
        for i in range(len(charging_indices)-1):
            current_indices = (df.index >= charging_indices[i]) & (df.index < charging_indices[i+1])
            df.loc[current_indices, 'testpart'] = testparts[i]

        return df

    def _identify_constant_charging_indices(self, df):
        constant_voltage_threshold = df['Voltage(V)'] > 3.599
        constant_voltage_condition = constant_voltage_threshold.rolling(window=100).sum() == 100
        voltage_drop_condition = df['Voltage(V)'].shift(-1) < 3.599
        charging_end_indices = df.index[constant_voltage_condition & voltage_drop_condition]

        current_condition = df['Current(A)'] > 0.02
        current_flips = current_condition.ne(current_condition.shift(-1))
        current_flips_indices = df.index[current_flips]

        charging_start_indices = []
        flips_list = current_flips_indices.tolist()  # Just once outside the loop

        for end_idx in charging_end_indices:
            idx_pos = np.searchsorted(flips_list, end_idx)
            if idx_pos > 0:
                charging_start_indices.append(flips_list[idx_pos - 1] + 1)

        return charging_start_indices, charging_end_indices

    def soc_calculation(self,
                        df: pd.DataFrame,
                        time_col: str ="Test_Time(s)",
                        current_col: str = 'Current(A)') -> pd.DataFrame:

        capacity = self.capacity_calculation(df)
        discharging_phases = ['DST', 'US06', 'FUD']
        df['eta'] = float(1)
        df.loc[df['testpart'].isin(discharging_phases), 'eta'] = 0.995

        soc = np.cumsum(df[current_col] / df['eta'] * df[time_col].diff().fillna(0)) / capacity / 3600
        df['soc'] = soc
        # calibrating the end of the charging cycles to be 1 soc
        _, charging_end_indices = self._identify_constant_charging_indices(df)

        charging_end_socs = df.loc[charging_end_indices, 'soc']
        mean_charging_end_soc = charging_end_socs.mean()
        df['soc'] = df['soc'] - mean_charging_end_soc + 1
        return df

    def capacity_calculation(self,
                             df: pd.DataFrame,
                             time_col: str="Test_Time(s)",
                             current_col: str="Current(A)") -> float:

        charging_phases = ['charging_1', 'charging_2', 'charging_3']

        charging_phase_capacity = np.zeros([len(charging_phases)])
        for i, charging_phase in enumerate(charging_phases):
            charging_phase_df = df[df['testpart'] == charging_phase].copy()
            time_diff = charging_phase_df[time_col].diff().fillna(charging_phase_df[time_col].diff().iloc[1])
            coulumb_counting = np.cumsum(charging_phase_df[current_col] * time_diff) / 3600
            charging_phase_df.loc[:, 'coulumb_counting'] = coulumb_counting
            charging_phase_capacity[i] = charging_phase_df['coulumb_counting'].iloc[-1] - charging_phase_df['coulumb_counting'].iloc[0]

        capacity = charging_phase_capacity.mean()
        return capacity