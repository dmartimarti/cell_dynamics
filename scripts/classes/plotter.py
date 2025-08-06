import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np
import os

class Plotter:
    def __init__(self):
        pass

    def plot_individual_plate_plotly(self, data, title, out_name, time_h, save=False):
        """
        Plots the data as a grid of 8x12 using plotly. It takes a Pandas dataframe as input with Wells as index and time as columns.

        Parameters
        ----------
        data : Pandas dataframe
            data to plot
        title : str
            title of the plot
        out_name : str
            name of the output file
        time_h : numpy array
            time in hours (e.g., [0., 0.5, 1.0, ...])
        save : bool, optional
            if True, saves the plot as a pdf file, by default False

        Returns
        -------
        plotly figure
        """

        # get index from data and separate it into numbers and letters, save only a unique list of both
        index = data.index.to_list()
        letters = list(set([i[0] for i in index]))
        numbers =list(set([int(i[1:]) for i in index]))

        # sort the lists
        letters.sort()
        numbers.sort()

        # max of y axis
        max_y = data.max(axis=1).max()

        # steps between 0 and max, rounded to 1 decimal
        step = round(max_y/3, 1)
        # this solves a bug when step is too small to be rounded to 1 decimal
        if step == 0.0:
            step = round(max_y/3, 2)

        # max x axis
        max_x = data.columns.max()
        step_x = round(max_x/2, 0)

        let_len = len(letters)
        num_len = len(numbers)

        fig = make_subplots(rows=let_len, cols=num_len,
                            shared_xaxes=True,
                            shared_yaxes=True,
                            subplot_titles=numbers,
                            vertical_spacing=0.03,
                            horizontal_spacing=0.011)

        for i, row in enumerate(letters):
            for j, col in enumerate(numbers):
                # if a combination of row and column is not in the dataframe, skip it
                if f'{row}{col}' not in data.index:
                    continue
                else:
                    # set color to black and make plots wider
                    fig.add_trace(go.Scatter(x=time_h,
                                            y=data.loc[f'{row}{col}'],
                                            line=dict(color='black', width=1)),
                                            row=i+1, col=j+1)
                    # y axis update axes, range between min and max of the data
                    if j == 0:
                        fig.update_yaxes(range=[0, max_y], gridcolor='white',
                                        title_text=row,
                                        showline=True, linewidth=1, linecolor='black',mirror=True,
                                        row=i+1, col=j+1,
                                        tickvals=np.arange(0, max_y, step))
                    else:
                        fig.update_yaxes(range=[0, max_y], gridcolor='white',
                                        showline=True, linewidth=1, linecolor='black',mirror=True,
                                        row=i+1, col=j+1,
                                        tickvals=np.arange(0, max_y, step))
                    # x axis, rotate the tick labels by 90 degrees
                    fig.update_xaxes(range=[0, 24], row=i+1, col=j+1, gridcolor='white',
                                    showline=True, linewidth=1, linecolor='black', mirror=True,
                                    tickvals=np.arange(0, max_x+(max_x*0.1), step_x),
                                    ticktext=np.arange(0, max_x+(max_x*0.1), step_x), tickangle=0)


        # make individual subplots wider
        # rotate y title_text by 90 degrees
        fig.update_layout(title_text=title,
                    title_x=0.5, title_y=0.95, title_font_size=20,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, height=700, width=1000)

        # save the plot in pdf format
        if save:
            fig.write_image(f'{out_name}.pdf')

    def plotly_wrapper(self, time_data, plate, data_type, output):
        """
        Wrapper function to plot the timeseries data using plotly
        Parameters
        ----------
        time_data : pandas dataframe
            The timeseries data to be plotted
        plate : str
            The plate name
        data_type : str
            The data type
        time_h : list
            The time in hours
        Returns
        -------
        None.
        """

        ts = time_data[(time_data.File == plate) & (time_data.Data == data_type)]

        # check if the end of plate is = '.txt' and remove it
        if plate[-4:] == '.txt':
            plate = plate[:-4]

        # remove non-numeric columns
        time_h = [int(i) for i in time_data.columns if self.is_number(i)]

        ts_col = time_h.copy()
        ts_col.insert(0, 'Well')

        ts = ts[ts_col]
        ts = ts.set_index('Well')

        time_h = sorted(time_h)
        time_h = np.array(time_h)/60/60

        out_file = f'{output}/{plate}_{data_type}'

        self.plot_individual_plate_plotly(ts, plate + data_type, out_file, time_h = time_h, save=True)

    def is_number(self, s):
        """
        Function to check if a string is a number

        Parameters
        ----------
        s : str
            The string to be checked

        Returns
        -------
        bool
            True if the string is a number, False otherwise

        """
        try:
            int(s)
            return True
        except ValueError:
            return False

    def plot_boxplots(self, data, grouping_var, temp_var, x_var, y_var, output_dir):
        """
            Function to make boxplots from the dataframe

            Parameters
            ----------
            data : pandas dataframe
                dataframe containing the data to be plotted
            grouping_var : str
                main variable by which you want to plot the data (e.g. Strains)
            temp_var : str
                variable to be used as a temporary variable to group the data (e.g. the specific strain)
            x_var : str
                variable to be used as the x-axis (e.g. the metformin concentration)
            y_var : str
                variable to be used as the y-axis (e.g. the AUC)
            output_dir : str
                directory to save the plots

            Returns
            -------
            Saves a plot in the specified directory
        """

        data = data[data[grouping_var] == temp_var]
        sns.boxplot(x=x_var, y=y_var, data=data)
        sns.swarmplot(x=x_var, y=y_var, data=data, color='black')
        plt.title(temp_var)
        plt.savefig(output_dir + temp_var + '.pdf')
        plt.close()

    def plot_parametric_fit(self, x, y, model, params, title, output_path):
        plt.figure()
        plt.plot(x, y, 'o', label='data')
        plt.plot(x, model(x, *params), '-', label='fit')
        plt.legend()
        plt.title(title)
        plt.savefig(output_path)
        plt.close()

    def plot_pca(self, data, color_by, explained_variance, output_path):
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x='PC1', y='PC2', data=data, hue=color_by, palette='viridis')
        plt.title('PCA of growth curves')
        plt.xlabel(f'PC1 ({explained_variance[0]:.2f})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2f})')
        plt.savefig(os.path.join(output_path, 'PCA_plot.pdf'))
        plt.close()
