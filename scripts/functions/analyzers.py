# analyzers.py
import numpy as np
from scipy import signal
from scipy import interpolate as ip
from itertools import product
import os
import pandas as pd
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning


class GrowthAnalyzer:
    def __init__(self, design, root):
        self.design = design
        self.root = root

    def biospa_text_opener(self, file):
        """
        Main function for opening the txt files from biospa
        :param file: a path to the txt file corresponding to a single plate in your computer
        :return: a tuple of a pandas dataframe, a vector of temperatures, and an OD value for the plate
        """
        # opens the file
        with open(file, 'r', errors='ignore') as f:
            contents = f.readlines()
        # save the OD
        od = contents[0].split('\n')[0]

        # get the index of the line where the times start
        for i, line in enumerate(contents):
            if line[:4] == 'Time':
                time_line = i
                break

        # save the times
        times = contents[time_line].split('\t')[1:-1]
        times = self.fix_times(times)  # this fixes size and values for the times, rounding them
        # save temperatures
        temps_raw = contents[3].split('\t')[1:-1]
        temps = [float(temp) for temp in temps_raw if temp not in ['0.0', '']]
        temps = np.array(temps)
        # save the useful data info
        temp_df = contents[4:len(contents)]
        # convert it to a pandas object
        df = self.df_beautify(temp_df, times=times)

        return df, temps, od

    def df_beautify(self, txt_object, times):
        """
        Function to modify pandas dataframes coming from the function biospa_text_opener
        and get them ready to be analysed
        :param times: the times vector from the txt file
        :param txt_object: the txt object that is being read by biospa_txt_opener
        :return: a better pandas dataframe
        """
        df = pd.DataFrame([x.split('\t') for x in txt_object])
        df = df.set_index(df[0])  # set index as well name
        df = df.drop(0, axis=1)
        df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)  # remove last column as it's a \n
        df = df.dropna()  # remove empty rows
        df = df.replace(r'^\s*$', np.nan, regex=True)  # replace empty values (from 0:00:00 time values) for NaN
        df.dropna(axis=1, inplace=True)  # remove empty columns
        df.columns = times  # put times as column names
        df = df.apply(pd.to_numeric)  # change type to numeric

        # check if we have missing wells in the dataframe and add them if needed
        # set of letters from A to H
        # get the index and separate letters and numbers, save the unique instances
        df_index = df.index
        letters = [i[0] for i in df_index]
        numbers = [i[1:] for i in df_index]
        # sort the list of letters and numbers
        letters = sorted(list(set(letters)))
        numbers = sorted(list(set(numbers)))
        combinations = self.get_well_names(len(letters), len(numbers))

        # check if there is a missing well, and if so, add it with 0 values
        if len(df.index) != len(combinations):
            missing_wells = list(set(combinations) - set(df.index))
            for well in missing_wells:
                df.loc[well] = 0
        else:
            pass

        # sort the dataframe by index following the order of the combinations
        df = df.reindex(combinations)

        return df

    def check_outliers_temps(self, temp_vect, thres: float = 0.1):
        """
        Checks if there are outliers in the temperature vector
        :param temp_vect: a numpy array of temperatures
        :param thres: the threshold to detect outliers, in proportion of the average value (0.1 by default)
        :return: a boolean, and the numpy array of the possible outliers
        """
        temps_av = np.average(temp_vect) * thres
        min_temp, max_temp = np.average(temp_vect) - temps_av, np.average(temp_vect) + temps_av
        filter_temps = temp_vect[np.logical_and(temp_vect > max_temp, temp_vect < min_temp)]
        if len(filter_temps) == 0:
            return False
        else:
            return True, filter_temps

    def get_time_h(self, df):
        # time_t is the time from the experiment in format 0:00:00
        time_t = df.columns.to_list()
        time_t = [t+52 for t in time_t]
        # length is the number of time points
        length = len(time_t)
        # timestep is the time between each time point
        timestep = self.round_to(float(time_t[-1] - time_t[0]) / (length - 1), 1)
        # timemax_min is the total time in minutes
        timemax_min = int((length - 1) * timestep / 60)  # time max in mins
        # timemax_remmin is the remainder of minutes
        timemax_h, timemax_remmin = divmod(timemax_min, 60)
        # time_span is the time in seconds
        time_span = np.linspace(0, timemax_min * 60, length, dtype=np.dtype(int))
        # time_h is the time in hours
        time_h = time_span / 3600.0
        return length, time_h, time_span

    def fix_times(self, times):
        """
        Takes a vector with the raw times values from the txt and fixes them into a proper shape and rounded values
        :param times: vector of times from the txt file from biospa
        :return: a vector of fixed times
        """
        times = [self.time_to_sec(val) for val in times if val not in ['Time', '']]
        length = len(times)
        timestep = self.round_to(float(times[-1] - times[0]) / (length - 1), 1)
        timemax_min = int((length - 1) * timestep / 60)  # time max in minutes
        time_span = np.linspace(0, timemax_min * 60, length, dtype=np.dtype(int))
        return time_span

    def get_well_names(self, num_letters, num_numbers):
        """
        Function that returns a list of all possible combinations of letters (from A to a specified number) and numbers (from 1 to a specified number)
        Parameters
        ----------
        num_letters : int
            The number of letters to be used
        num_numbers : int
            The number of numbers to be used
        Returns
        -------
        list
            A list of all possible combinations of letters (from A to a specified number) and numbers (from 1 to a specified number)
        """
        # get the letters
        letters = [chr(i) for i in range(65, 65+num_letters)]
        # get the numbers
        numbers = [str(i) for i in range(1, num_numbers+1)]
        # get all the combinations of letters and numbers
        well_names = [i+j for i, j in product(letters, numbers)]
        return well_names

    def round_to(self, n, precision):
        """
        Function from the original script to round numbers to a desired precision
        (Need to check if I really need this)
        :param n: a float
        :param precision: an integer
        :return:
        """
        # Round a number to desired precision
        correction = 0.5 if n >= 0 else -0.5
        return int(n / precision + correction) * precision

    def time_to_sec(self, time_str):
        """
        Converts a time string from reading the file with biospa_text_opener and converts it to seconds
        :param time_str: a time string like '0:00:30'
        :return: an integer of the total seconds (30 in the case of the example)
        """
        h, m, s = time_str.split(':')
        seconds = int(s) + 60 * int(m) + 3600 * int(h)
        return seconds

    def set_series2min(self, x, thres: float = 0.0):
        """
        This function takes a pandas series and clip the values between 0 and Inf
        :param x: a Pandas Series object
        :param thres: threshold to use as a minimum value
        :return: returns a numpy array without negative values
        """
        x = x.to_numpy()
        x = np.clip(x, thres, np.inf)
        return x

    @staticmethod
    def gompertz_model(t, A, mu, t_lag):
        return A * np.exp(-np.exp(mu * np.exp(1) / A * (t_lag - t) + 1))

    def fit_gompertz(self, t, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", OptimizeWarning)
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                try:
                    params, _ = curve_fit(self.gompertz_model, t, y, p0=[max(y), 0.1, 1], maxfev=10000)
                    return params
                except RuntimeError:
                    return [np.nan, np.nan, np.nan]

    def calculate_auc(self, file, mode):
        """
        Calculates the AUC of the OD time series.
        """
        # open the info about df, temperatures, times and OD
        df, temps, OD = self.biospa_text_opener(os.path.join(self.root, file))
        OD = OD[3:]

        if self.check_outliers_temps(temps):
            raise Exception("There are outliers in the temperatures, check your experiment!")
        else:
            pass

        # get time info
        length, time_h, time_span = self.get_time_h(df)

        ### Fix and interpolate the data
        window = int(round(length / 10, 0))
        # make all the time series start from the same point (0 usually)
        adj_df = df.apply(lambda row: pd.Series(self.set_series2min(row - np.mean(row[:window]), 0.0), index=time_span), axis=1)

        # smooth the data from adj_df using a wiener filter
        with np.errstate(divide='ignore', invalid='ignore'): # ignore the warnings from wiener filter
            w_filt = adj_df.apply(lambda row: pd.Series(signal.wiener(row, 5), index=time_span), axis=1)
        # fix the NaN rows and substitute them with 0
        w_filt = w_filt.fillna(0)

        growth_rates = w_filt.apply(lambda row: pd.Series(ip.UnivariateSpline(time_h, row, s=0).derivative(1)(time_h), index=time_span),
                        axis=1)

        max_slope = growth_rates.apply(lambda row: pd.Series(row.rolling(window).mean().max(), index=time_span), axis=1).iloc[:,0]

        # calculate AUC from the smoothed data
        auc = w_filt.apply(lambda row: pd.Series(np.trapz(row, time_h), index=time_span), axis=1).iloc[:,0]

        ######## prepare the Output.csv
        # remove RuntimeWarning: divide by zero encountered in log2
        with np.errstate(divide='ignore', invalid='ignore'):
            auc_log2 = np.log2(auc)

        gompertz_params = w_filt.apply(lambda row: self.fit_gompertz(time_h, row), axis=1)
        gompertz_df = pd.DataFrame(gompertz_params.to_list(), columns=['A', 'mu', 't_lag'], index=w_filt.index)

        auc_df = pd.DataFrame({f'File': file, f'{OD}_f_AUC': auc, f'{OD}_f_logAUC': auc_log2, f'{OD}_dt_Max': max_slope})
        auc_df.reset_index(inplace=True)
        auc_df = auc_df.rename(columns = {0:'Well'})

        # Merge with Gompertz parameters
        auc_df = auc_df.merge(gompertz_df, left_on='Well', right_index=True)

        if mode == 'AUC':
            return auc_df
        elif mode == 'timeseries':
            return self.smooth_data(file)

    def smooth_data(self, file):
        """
        Smooths the data using a Wiener filter.
        """
        # open the info about df, temperatures, times and OD
        df, temps, OD = self.biospa_text_opener(os.path.join(self.root, file))
        OD = OD[3:]

        if self.check_outliers_temps(temps):
            raise Exception("There are outliers in the temperatures, check your experiment!")
        else:
            pass

        # get time info
        length, time_h, time_span = self.get_time_h(df)

        ### Fix and interpolate the data
        window = int(round(length / 10, 0))
        # make all the time series start from the same point (0 usually)
        adj_df = df.apply(lambda row: pd.Series(self.set_series2min(row - np.mean(row[:window]), 0.0), index=time_span), axis=1)

        # smooth the data from adj_df using a wiener filter
        with np.errstate(divide='ignore', invalid='ignore'): # ignore the warnings from wiener filter
            w_filt = adj_df.apply(lambda row: pd.Series(signal.wiener(row, 5), index=time_span), axis=1)
        # fix the NaN rows and substitute them with 0
        w_filt = w_filt.fillna(0)

        growth_rates = w_filt.apply(lambda row: pd.Series(ip.UnivariateSpline(time_h, row, s=0).derivative(1)(time_h), index=time_span),
                        axis=1)

        # get the timeseries
        # add metadata to adj_df
        adj_df.insert(0, 'File', file)
        adj_df.insert(1, 'Data', f'{OD}nm')
        adj_df.insert(2, 'Well', adj_df.index)
        # add metadata to w_filt
        w_filt.insert(0, 'File', file)
        w_filt.insert(1, 'Data', f'{OD}nm_f')
        w_filt.insert(2, 'Well', w_filt.index)
        # add metadata to growth_rates
        growth_rates.insert(0, 'File', file)
        growth_rates.insert(1, 'Data', f'{OD}nm_dt')
        growth_rates.insert(2, 'Well', growth_rates.index)

        # concat the dfs
        timeseries_df = pd.concat([adj_df, w_filt, growth_rates], axis=0)

        return timeseries_df