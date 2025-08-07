from scipy import signal
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
import os

class GrowthAnalyzer:
    def __init__(self):
        pass

    def calculate_auc(self, file, root, mode):
        """
        Calculates the AUC of the OD time series.
        """
        # open the info about df, temperatures, times and OD
        from .processor import DataProcessor
        processor = DataProcessor()
        df, temps, OD = processor.biospa_text_opener(os.path.join(root, file))
        OD = OD[3:]

        if processor.check_outliers_temps(temps):
            raise Exception("There are outliers in the temperatures, check your experiment!")
        else:
            pass

        # get time info
        length, time_h, time_span = processor.get_time_h(df)

        ### Fix and interpolate the data
        window = int(round(length / 10, 0))
        # make all the time series start from the same point (0 usually)
        adj_df = df.apply(lambda row: pd.Series(processor.set_series2min(row - np.mean(row[:window]), 0.0), index=time_span), axis=1)

        # smooth the data from adj_df using a wiener filter
        with np.errstate(divide='ignore', invalid='ignore'): # ignore the warnings from wiener filter
            w_filt = adj_df.apply(lambda row: pd.Series(signal.wiener(row, 5), index=time_span), axis=1)
        # fix the NaN rows and substitute them with 0
        w_filt = w_filt.fillna(0)
        growth_rates = w_filt.apply(lambda row: pd.Series(UnivariateSpline(time_h, row, s=0).derivative(1)(time_h), index=time_span),
                        axis=1)

        max_slope = growth_rates.apply(lambda row: pd.Series(row.rolling(window).mean().max(), index=time_span), axis=1).iloc[:,0]

        # calculate AUC from the smoothed data
        auc = w_filt.apply(lambda row: pd.Series(np.trapz(row, time_h), index=time_span), axis=1).iloc[:,0]

        ######## prepare the Output.csv
        # remove RuntimeWarning: divide by zero encountered in log2
        with np.errstate(divide='ignore', invalid='ignore'):
            auc_log2 = np.log2(auc)

        auc_df = pd.DataFrame({f'File': file, f'{OD}_f_AUC': auc, f'{OD}_f_logAUC': auc_log2, f'{OD}_dt_Max': max_slope})
        auc_df.reset_index(inplace=True)
        auc_df = auc_df.rename(columns = {0:'Well'})

        if mode == 'AUC':
            return auc_df
        elif mode == 'timeseries':
            return processor.smooth_data(file, root)

    def growth(self, x, A, lam, u):
        """
        Parametric logistic growth model.
        Ref: https://www.jstatsoft.org/article/download/v033i07/367
        :param x: series values
        :param A: carrying capacity or max growth
        :param lam: length of lag phase
        :param u: growth rate
        :return: returns the model to be optimised with curve_fit from scipy
        """
        return A / (1 + np.exp((4 * u / A) * (lam - x) + 2))

    def gompertz(self, x, A, lam, u):
        """
        Gompertz growth model.
        Ref: https://www.jstatsoft.org/article/download/v033i07/367
        :param x: series values
        :param A: carrying capacity or max growth
        :param lam: length of lag phase
        :param u: growth rate
        :return: returns the model to be optimised with curve_fit from scipy
        """
        return A * np.exp(-np.exp(u * np.exp(1) / A * (lam - x) + 1))

    def calculate_aic(self, n, sse, k):
        """
        Calculates the Akaike Information Criterion (AIC).
        :param n: number of data points
        :param sse: sum of squared errors
        :param k: number of parameters
        :return: AIC value
        """
        if sse == 0:
            return -np.inf
        else:
            return n * np.log(sse / n) + 2 * k

    def model_selection(self, x, y, models):
        """
        Selects the best model based on AIC.
        :param x: x data
        :param y: y data
        :param models: dictionary of models to be tested
        :return: best model name, parameters and aic
        """
        best_model = None
        best_aic = np.inf
        best_params = None

        for name, model in models.items():
            try:
                params, _ = curve_fit(model, x, y)
                y_fit = model(x, *params)
                sse = np.sum((y - y_fit) ** 2)
                aic = self.calculate_aic(len(y), sse, len(params))

                if aic < best_aic:
                    best_aic = aic
                    best_model = name
                    best_params = params
            except RuntimeError:
                pass

        return best_model, best_params, best_aic

    def run_pca(self, data, color_by):
        """
        Performs PCA on the data.
        """
        # get the data to be used in the PCA
        pca_data = data[data.columns[4:]].values

        # perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_data)

        # create a dataframe with the PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

        # merge with the metadata
        pca_df = pd.concat([data.reset_index(drop=True), pca_df], axis=1)

        return pca_df, pca.explained_variance_ratio_
