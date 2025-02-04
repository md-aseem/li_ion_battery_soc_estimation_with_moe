import time
from config import DataParams
from utils.preprocess_data import PreprocessData

if __name__ == "__main__":

    data_path = 'data/Stage_1'
    data_params = DataParams()
    read_process_data = PreprocessData(data_params)
    start_time = time.time()
    df = read_process_data.load_dfs(data_path)
    df = read_process_data.add_sequence_data(df, 'c_cur', num_points=5)
    print(f"Total Time to Load: {(time.time() - start_time):0f} seconds")
    print(f"Shape of df: {df.shape}")

    read_process_data.plot(df=df,
                           y_axes=['c_vol', 'soc'],
                           conditions={'testpoint':2},
                           downsampling_factor=10)
