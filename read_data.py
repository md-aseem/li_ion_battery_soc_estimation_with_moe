from utils.helper_functions import plot_data
import time
from utils.params import DataParams
from utils.data_import import import_datafile
from utils.helper_functions import ReadData

if __name__ == "__main__":

    data_path = 'data/Stage_1'
    data_params = DataParams()
    read_data = ReadData(data_params)
    start_time = time.time()
    df = read_data.load_dfs(data_path)

    print(df.head())
    print(f"Total Time: {time.time() - start_time}")
    print(df.shape)
    df['time_res'] = df['run_time'].diff()

    plot_data(df, y_axes=['test_parts', 'c_vol'], condition_col='testpoint', condition_value=4)
