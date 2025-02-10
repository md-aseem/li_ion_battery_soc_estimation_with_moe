import time
from config import MultiStageDataParams
from utils.preprocess_data import PreprocessMultiStageData

if __name__ == "__main__":

    data_path = 'data/Stage_1'
    data_params = MultiStageDataParams()
    read_process_data = PreprocessMultiStageData(data_params)
    start_time = time.time()
    df = read_process_data.load_dfs(data_path)
    df = read_process_data.add_sequence_data(df, 'c_cur', history_length=data_params.num_points)
    feature_cols = ['c_cur', 'soc', 'ocv', 'dva'] + [f"c_cur-{i+1}" for i in range(data_params.num_points)]
    df, _ = read_process_data.standardize_data(df, feature_cols=feature_cols)
    print(f"Total Time to Load: {(time.time() - start_time):0f} seconds")
    print(f"Shape of df: {df.shape}")

    read_process_data.plot(df=df,
                           y_axes=['c_vol', 'soc'],
                           conditions={'testpoint':2},
                           downsampling_factor=10)
