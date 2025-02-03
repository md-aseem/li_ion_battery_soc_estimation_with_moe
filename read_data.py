import time
from utils.params import DataParams
from utils.helper_functions import ReadAndProcessData

if __name__ == "__main__":

    data_path = 'data/Stage_1'
    data_params = DataParams()
    read_process_data = ReadAndProcessData(data_params)
    start_time = time.time()
    df = read_process_data.load_dfs(data_path)
    print(df.head())
    print(f"Total Time: {time.time() - start_time}")
    print(df.shape)

    read_process_data.plot_data(df, y_axes=['test_parts', 'c_vol'], conditions={'testpoint':4} )
