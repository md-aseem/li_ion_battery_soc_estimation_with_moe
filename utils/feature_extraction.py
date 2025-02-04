import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

class CapacityAndSOCCalculation:
    def __init__(self):
        pass

    def add_capacity_soc_cols(self, df: pd.DataFrame):
        capacity_dict = self.calc_capacity(df)
        Q_mean = capacity_dict['Q_mean']; Q_ch = capacity_dict['Q_ch']; Q_dch = capacity_dict['Q_dch']
        df['Q_mean'] = Q_mean; df['Q_ch'] = Q_ch; df['Q_dch'] = Q_dch
        df['soc'] = self._q_calc(df) / df['Q_mean'] / 3600
        zero_soc_index = df[df['step_type'] == 21].index[0] # selecting first timestep of pocv cycle as zero soc value
        df['soc'] = df['soc'] - df.loc[zero_soc_index, 'soc']
        return df

    def calc_capacity(self, df: pd.DataFrame) -> dict:
        """
        Evaluate capacity of measurement in df with step_type = [21,22]

        :param df: pd.DataFrame: df with step_type = [21,22]
        :return: {'Q_mean': float, 'Q_ch': float, 'Q_dch': float, 'q_ch': np.ndarray, 'q_dch': np.ndarray}
        """
        q_calc = self._q_calc
        q_kapa_ch = q_calc(df[(df.step_type == 21) & (df.c_cur > 0)])
        q_kapa_dch = q_calc(df[(df.step_type == 22) & (df.c_cur < 0)])

        capa_ch = q_kapa_ch[-1] - q_kapa_ch[0]
        capa_dch = q_kapa_dch[0] - q_kapa_dch[-1]

        capa_mean = (capa_ch + capa_dch) / 2 / 3600
        capa_ch = capa_ch / 3600
        capa_dch = capa_dch / 3600

        return {'Q_mean': capa_mean, 'Q_ch': capa_ch, 'Q_dch': capa_dch, 'q_ch': q_kapa_ch, 'q_dch': q_kapa_dch}

    def get_df_capacity(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracting the capacity measurement from larger DataFrame

        :param df:
        :return df_capa: pd.DataFrame: DataFrame containing capacity measurement only
        """
        df_capa_meas = df[((df.step_type == 21) & (df.c_cur > 0)) | ((df.step_type == 22) & (df.c_cur < 0))]

        return df_capa_meas

    def _q_calc(self, df: pd.DataFrame) -> np.ndarray:
        """
        Integrate current over time to get charge throughput

        :param df:
        :return q: np.array:
        """
        q = np.cumsum((df['run_time'].diff() * df['c_cur']).fillna(0).values)

        return q



class OCVDVACalculation:
    def __init__(self):
        pass

    def add_ocv_dva(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracting OCV and DVA and then merging them on each row.
        :param df: Raw Df containing c_cur and SOC columns
        :return: df: pd.DataFrame: DataFrame containing DVA and OCV columns attached
        """
        assert 'soc' in df.columns.to_list(), "'soc' column is required in the dataframe."

        soc_ocv_dict = self.calc_ocv_curve(df=df)
        soc = soc_ocv_dict['SOC']; ocv = soc_ocv_dict['OCV']; ocv_ch = soc_ocv_dict['OCV_ch']; ocv_dch = soc_ocv_dict['OCV_dch']

        dva = self.calc_dva_curve(ocv, soc, smooth=True)
        dva_ch = self.calc_dva_curve(ocv_ch, soc, smooth=True)
        dva_dch = self.calc_dva_curve(ocv_dch, soc, smooth=True)

        ocv_dva_df = pd.DataFrame({"soc": soc/100,
                                   "ocv": ocv,
                                   "ocv_ch": ocv_ch,
                                   "ocv_dch": ocv_dch,
                                   "dva": dva,
                                   "dva_ch": dva_ch,
                                   "dva_dch": dva_dch})

        df['soc'] = df['soc'].round(3); ocv_dva_df['soc'] = ocv_dva_df['soc'].round(3) # for perfect join
        df = pd.merge(df, ocv_dva_df, on='soc', how='left')

        for col in ocv_dva_df.columns:
            df[col] = df[col].interpolate()

        return df
    def get_df_ocv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracting the ocv measurement from larger DataFrame

        :param df: pd.DataFrame: larger dataframe containing ocv measurement
        :return df_ocv_meas: pd.DataFrame: DataFrame containing ocv measurement only
        """

        df_ocv_meas = df[(df.step_type == 31) | (df.step_type == 32)]

        return df_ocv_meas


    def calc_ocv_curve(self, df: pd.DataFrame, enable_filter: bool = False, polyorder: int = 3) -> dict:
        """
        Extracting the open circuit voltage curve (OCV) contained in the dataframe with step_type = [31,32].
        Here, only the constant current CC-OCV measurement can be evaluated.

        :param polyorder: Order of the polynomial filter for smoothing the OCV curve
        :type polyorder: int
        :param enable_filter: Enable filter for smoothing the OCV curve
        :type enable_filter: bool
        :param df:
        :return: {'SoC': np.ndarray, 'OCV': np.ndarray, 'OCV_ch': np.ndarray, 'OCV_dch': np.ndarray}
        """
        # Select the charging (CC) / discharging areas and store as vectors
        c_cur_dch = df.c_cur[(df.step_type == 32)].values
        c_vol_dch = df.c_vol[(df.step_type == 32)].values

        c_cur_ch = df.c_cur[(df.step_type == 31)].values
        c_vol_ch = df.c_vol[(df.step_type == 31)].values

        # Time section division into soc-interval
        soc_dch = np.transpose(np.linspace(100, 0, (len(c_cur_dch))))
        soc_ch = np.transpose(np.linspace(0, 100, (len(c_cur_ch))))

        # Interpolation of the charging and discharging curve
        soc = np.linspace(0, 100, 1001)
        ocv_dch = griddata(soc_dch, c_vol_dch, soc, method="nearest")
        ocv_ch = griddata(soc_ch, c_vol_ch, soc, method="nearest")

        if enable_filter:
            window_size = int(len(ocv_dch) * 0.001)
            ocv_dch = savgol_filter(ocv_dch, window_size, polyorder)
            ocv_ch = savgol_filter(ocv_ch, window_size, polyorder)

        # Formation of the open circuit voltage curve (average)
        ocv = (ocv_dch + ocv_ch) / 2

        return {'SOC': soc.round(2), 'OCV': ocv.round(4), 'OCV_ch': ocv_ch.round(4), 'OCV_dch': ocv_dch.round(4)}


    def calc_dva_curve(self, ocv: np.ndarray, soc: np.ndarray, smooth: bool = False) -> np.ndarray:
        """
        Differentiate the OCV-curve to get the DVA-curve

        :param ocv: Array of OCV values
        :type ocv: np.ndarray
        :param soc: Array of SoC values
        :type soc: np.ndarray
        :param smooth: Whether to smooth the dOCVdSOC curve or not
        :type smooth: bool
        :return: dva
        """
        dva = np.gradient(ocv, soc)
        if smooth:
            dva = savgol_filter(dva, window_length=10, polyorder=3)

        return dva.round(6)