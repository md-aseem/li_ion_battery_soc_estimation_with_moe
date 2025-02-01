import matplotlib.pyplot as plt
from utils.data_import import import_datafile
import numpy as np
import os
import zipfile
import re
import pandas as pd

def plot_data(df, x_axis= 'run_time', y_axes:list = None, condition_col=None, condition_value=None):

    if y_axes is None:
        y_axes = ['c_cur', 'c_vol']
    if condition_value and condition_col:
        plot_df = df[df[condition_col] == condition_value]
    else:
        plot_df = df

    for y_axis in y_axes:
        plt.plot(plot_df[x_axis], plot_df[y_axis], label=y_axis)
    plt.xlabel(str(x_axis))
    plt.legend()
    plt.show()


class ReadData:
    def __init__(self, data_params):
        self.data_params = data_params
        self.relevant_csv_list = self.provide_relevant_filenames(data_params)
        self.df = None

    def load_dfs(self, data_path):

        relevant_filenames = self.relevant_csv_list
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
                            df = self.identify_test_part(df)
                            filenames.append(filename)
                            df_list.append(df)

        print(f"The number of files loaded: {len(df_list)}\n",
              f"File Names are: {filenames}")

        self.df = pd.concat(df_list, axis=0)
        return self.df

    def provide_relevant_filenames(self, data_params):

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
        self.relevant_csv_list = relevant_csv_list
        return relevant_csv_list

    def extract_parameters(self, file_name):

        file_name = file_name.split('/')[-1]
        file_name_list = file_name.split("_")

        aging_type =re.sub(r'[^a-zA-Z]', '', file_name_list[1])
        testpoint = re.sub(r'[^0-9]', '', file_name_list[1])
        rpt = file_name_list[-2]
        temperature = re.sub(r'[^0-9]', '', file_name_list[-1])

        return aging_type, testpoint, rpt, temperature

    def identify_test_part(self, df):

        pocv_starting_index = df[df['step_type'] > 30].index[0]
        hppc_starting_index = df[df['step_type'] > 100].index[0]

        conditions = [
            (df.index < pocv_starting_index),
            (df.index >= pocv_starting_index) & (df.index <= hppc_starting_index),
            (df.index > hppc_starting_index)
        ]

        test_parts = ['full_charge_discharge', 'pocv', 'hppc']

        df['test_parts'] = np.select(conditions, test_parts, default="NA")

        return df




