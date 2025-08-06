#!/usr/bin/env python3
#
#     Copyright (C) 2022, Daniel Martínez Martínez
#

"""
Python script to transform the output data from the Biospa machine into
a format that can be treated with any other programming language.
The main idea comes from the script Growth.py from Povilas repository,
but this implementation is entirely mine.
"""

__author__ = 'Daniel Martínez Martínez'
__copyright__ = 'Copyright (C) 2022 Daniel Martínez Martínez'
__license__ = 'MIT License'
__email__ = 'dmartimarti **AT** gmail.com'
__maintainer__ = 'Daniel Martínez Martínez'
__status__ = 'beta'
__date__ = 'Dec 2024'
__version__ = '0.9.0'

from classes.reader import ExperimentReader
from classes.processor import DataProcessor
from classes.analyzer import GrowthAnalyzer
from classes.plotter import Plotter
import argparse
import pathlib
import os
import shutil
from itertools import product
from multiprocessing import get_context
from tqdm import tqdm
import pandas as pd


# class of colors to print in the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    parser = argparse.ArgumentParser()

    # define arguments to pass to the script
    # here comes one difference, I'll read a path instead of a Design file
    parser.add_argument('-i',
                        '--input',
                        type=lambda p: pathlib.Path(p).absolute(),
                        help='Input path where the design file is located (in .xlsx format)',
                        required=True)
    parser.add_argument('-f',
                        '--file',
                        default='Design.xlsx',
                        help='Name of the input file. "Design.xlsx" by default.')
    parser.add_argument('-o',
                        '--output',
                        default='Output',
                        help='Output folder to save the analysis')
    parser.add_argument('-t',
                        '--threads',
                        default=1,
                        help='Number of threads to use. 1 by default.')
    # add an argument to specify if you are using windows, optional
    parser.add_argument('-w',
                        '--windows',
                        action='store_true',
                        help='Flag to indicate that you are using Windows.')
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=f'%(prog)s v.{__version__} {__status__}')
    parser.add_argument('--parametric',
                        action='store_true',
                        help='Flag to enable parametric modelling.')
    parser.add_argument('--pca',
                        action='store_true',
                        help='Flag to enable PCA analysis.')
    parser.add_argument('--pca_color',
                        type=str,
                        default=None,
                        help='Variable to color the PCA plot by.')

    args = parser.parse_args()

    # initialise variables
    ROOT = args.input
    OUTPUT = args.output
    n_threads = int(args.threads)

    print(f'The folder {os.path.split(args.input)[-1]} will be analysed.')

    # if -w is passed, store a variable with the string "spawn"
    if args.windows:
        start_method = 'spawn'
    else:
        start_method = 'fork'

    # Initialize classes
    reader = ExperimentReader(os.path.join(ROOT, args.file))
    processor = DataProcessor()
    analyzer = GrowthAnalyzer()
    plotter = Plotter()

    # read Excel file with Pandas
    design = reader.read_design()

    # first, check if the files from the file and your computer are the same
    des_files = design['File'].to_list()
    # files_in_system = file_parser(path=ROOT, pattern='*.txt') # This function is not defined in the new structure
    files_in_system = [f for f in os.listdir(ROOT) if f.endswith('.txt')]


    print(f'Len of files in system: {len(files_in_system)}, and len of files in design: {len(des_files)}')

    # if set(des_files).issubset(set(files_in_system)): # This function is not defined in the new structure
    if set(des_files).issubset(set(files_in_system)):
        pass
    else:
        raise Exception("The files in the folder are not the same as "
                        f"the files within your {bcolors.BOLD}Design file!{bcolors.END}\n"
                        f"{bcolors.WARNING}Exiting from the run.{bcolors.ENDC}")

    # from this point, I need to open the files, calculate their stuff, plot them, and save the relevant
    # create an output folder, overwrite if it exists
    if not os.path.exists(os.path.join(ROOT, OUTPUT)):
        os.makedirs(os.path.join(ROOT, OUTPUT))
    else:
        # give option to overwrite or not
        overwrite = input(f'The folder {bcolors.OKBLUE}{OUTPUT}{bcolors.ENDC} already exists. Do you want to overwrite it? (y/n): \n')
        if overwrite == 'y':
            shutil.rmtree(os.path.join(ROOT, OUTPUT))
            os.makedirs(os.path.join(ROOT, OUTPUT))
        else:
            print(f'{bcolors.WARNING}Exiting from the run.{bcolors.ENDC}')
            exit()

    # create a folder for the plots
    if not os.path.exists(os.path.join(ROOT, OUTPUT, 'Plots')):
        os.makedirs(os.path.join(ROOT, OUTPUT, 'Plots'))
    else:
        # give option to overwrite or not
        overwrite = input(f'The folder {bcolors.OKBLUE}{OUTPUT}/Plots{bcolors.ENDC} already exists. Do you want to overwrite it? (y/n): \n')
        if overwrite == 'y':
            shutil.rmtree(os.path.join(ROOT, OUTPUT, 'Plots'))
            os.makedirs(os.path.join(ROOT, OUTPUT, 'Plots'))
        else:
            print('Plots will be saved in the existing folder.')

    # LOOP OVER THE FILES
    print(f'{bcolors.OKCYAN}Starting the analysis of the files: {bcolors.ENDC}\n')
    # parallel loop to get the AUCs
    print(f'{bcolors.OKCYAN}Calculating the AUCs...{bcolors.ENDC}\n')
    with get_context(start_method).Pool(n_threads) as p:
        # user starmap to pass multiple arguments to the function
        out_auc_df = pd.concat(list(tqdm(p.starmap(
                                                analyzer.calculate_auc,
                                                zip(design.File.to_list(),
                                                [ROOT]*len(design.File.to_list()),
                                                ['AUC']*len(design.File.to_list()))),
                                    total=len(design.File.to_list()))), axis=0)
    p.close()
    print('\n')
    # parallel loop to get the timeseries
    print(f'{bcolors.OKCYAN}Calculating the timeseries...{bcolors.ENDC}\n')
    with get_context(start_method).Pool(n_threads) as p:
        # user starmap to pass multiple arguments to the function
        out_time_df = pd.concat(list(tqdm(p.starmap(
                                                processor.smooth_data,
                                                zip(design.File.to_list(),
                                                [ROOT]*len(design.File.to_list()))),
                                    total=len(design.File.to_list()))), axis=0)
    p.close()
    print('\n')

    ### Pattern files
    if 'Pattern' in design.columns:
        patterns = design['Pattern'].unique().tolist()
        final_pattern_df = pd.DataFrame()
        for pattern in patterns:
            # pattern_vars = get_sheet_names(os.path.join(ROOT, pattern)) # This function is not defined in the new structure
            pattern_vars = pd.ExcelFile(os.path.join(ROOT, pattern)).sheet_names
            pattern_df = pd.DataFrame()
            for pat in pattern_vars:
                pattern_media = pd.read_excel(os.path.join(ROOT, pattern), pat)
                pattern_media = pattern_media.set_index(pattern_media.columns[0])
                well = processor.get_well_names(pattern_media.shape[0], pattern_media.shape[1])
                pattern_media_vec = pattern_media.values.flatten().tolist()
                pattern_media_df = pd.DataFrame({'Well': well, f'{pat}': pattern_media_vec, 'Pattern': pattern})
                pattern_df = pd.concat([pattern_df, pattern_media_df], axis=1)

            pattern_df = pattern_df.loc[:,~pattern_df.columns.duplicated()]
            pattern_df = pattern_df[['Pattern', 'Well'] + [col for col in pattern_df.columns if col not in ['Pattern', 'Well']]]
            final_pattern_df = pd.concat([final_pattern_df, pattern_df], axis=0)

    # merge the out_auc_df with the design
    # if 'Pattern', merge design with final_pattern_df and then merge with out_auc_df
    if 'Pattern' in design.columns:
        design_ext = design.merge(final_pattern_df, on=['Pattern'], how='right')
        out_auc_df = design_ext.merge(out_auc_df, on=['File', 'Well'], how='left')
    else:
        out_auc_df = design.merge(out_auc_df, on='File', how='left')
    # merge the out_time_df with the design
    # if 'Pattern', merge design with final_pattern_df and then merge with out_time_df
    if 'Pattern' in design.columns:
        design_ext = design.merge(final_pattern_df, on=['Pattern'], how='right')
        out_time_df = design_ext.merge(out_time_df, on=['File', 'Well'], how='left')
    else:
        out_time_df = design.merge(out_time_df, on='File', how='left')

    # save the output file in the output folder as a csv file
    out_auc_df.to_csv(os.path.join(ROOT, OUTPUT, 'Summary.csv'), index=False)
    # save the timeseries file in the output folder as a csv file
    out_time_df.to_csv(os.path.join(ROOT, OUTPUT, 'Timeseries.csv'), index=False)

    # PARAMETRIC MODELLING
    if args.parametric:
        print(f'\n{bcolors.OKCYAN}Starting the parametric modelling...{bcolors.ENDC}\n')

        # create a folder for the parametric plots
        parametric_path = os.path.join(ROOT, OUTPUT, 'Plots', 'Parametric')
        if not os.path.exists(parametric_path):
            os.makedirs(parametric_path)
        else:
            print('The Parametric folder already exists. I will overwrite the files.')

        models = {'logistic': analyzer.growth, 'gompertz': analyzer.gompertz}

        # get the timeseries data for the smoothed data
        time_h = [int(i) for i in out_time_df.columns if plotter.is_number(i)]
        time_h = [t/3600 for t in time_h]

        # get the smoothed data
        w_filt_df = out_time_df[out_time_df.Data.str.endswith('_f')]

        # create a new dataframe to store the results
        parametric_df = pd.DataFrame()

        for index, row in tqdm(w_filt_df.iterrows(), total=w_filt_df.shape[0]):
            y = row[w_filt_df.columns[4:]].values
            x = time_h

            best_model, best_params, best_aic = analyzer.model_selection(x, y, models)

            if best_params is not None:
                # create a dictionary with the results
                results = {'File': row['File'],
                           'Well': row['Well'],
                           'model': best_model,
                           'A': best_params[0],
                           'lam': best_params[1],
                           'u': best_params[2],
                           'aic': best_aic}
                # append the results to the dataframe
                parametric_df = parametric_df.append(results, ignore_index=True)

                # plot the results
                plotter.plot_parametric_fit(x, y, models[best_model], best_params,
                                            f'{row.File}_{row.Well}',
                                            os.path.join(parametric_path, f'{row.File}_{row.Well}.pdf'))

        # merge the parametric_df with the out_auc_df
        out_auc_df = out_auc_df.merge(parametric_df, on=['File', 'Well'], how='left')
        out_auc_df.to_csv(os.path.join(ROOT, OUTPUT, 'Summary.csv'), index=False)

    # PCA ANALYSIS
    if args.pca:
        print(f'\n{bcolors.OKCYAN}Starting the PCA analysis...{bcolors.ENDC}\n')

        # create a folder for the PCA plots
        pca_path = os.path.join(ROOT, OUTPUT, 'Plots', 'PCA')
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)
        else:
            print('The PCA folder already exists. I will overwrite the files.')

        # get the smoothed data
        w_filt_df = out_time_df[out_time_df.Data.str.endswith('_f')]

        pca_df, explained_variance = analyzer.run_pca(w_filt_df, args.pca_color)

        # plot the results
        plotter.plot_pca(pca_df, args.pca_color, explained_variance, pca_path)


    ### PLOT THE TIMESERIES
    data_types = out_time_df.Data.unique()
    plates = out_time_df.File.unique()
    out_path = os.path.join(ROOT, OUTPUT, 'Plots')
    print_out = os.path.join(OUTPUT,'Plots')

    print(f'\nPlotting the {bcolors.OKGREEN}timeseries{bcolors.ENDC} in {print_out}. \n')

    plotly_inputs = zip([out_time_df]*len(list(product(plates, data_types))),
                        [plate for plate in plates for i in range(len(data_types))],
                        [data_type for i in range(len(plates)) for data_type in data_types],
                        [out_path]*len(list(product(plates, data_types))))

    # print(out_path)
    # loop over the plates and data types using plotly_wrapper function
    with get_context(start_method).Pool(n_threads) as p:
        p.starmap(plotter.plotly_wrapper, tqdm(plotly_inputs, total=len(list(product(plates, data_types)))))

    # EXTENDED ANALYSIS
    # desing_sheets = get_sheet_names(os.path.join(ROOT, 'Design.xlsx')) # This function is not defined in the new structure
    desing_sheets = pd.ExcelFile(os.path.join(ROOT, 'Design.xlsx')).sheet_names
    if 'analysis' in desing_sheets:
        print('\nI found the analysis sheet in the design file. I will run the extended analysis.\n')
        analysis_vars = pd.read_excel(os.path.join(ROOT, 'Design.xlsx'), 'analysis')

        # analysis_vars = AnalysisVars(analysis_vars) # This class is not defined in the new structure
        grp_var = analysis_vars.grouping_variable.dropna().to_list()
        condition = analysis_vars.condition.dropna().to_list()[0]

        # check that grp_var and condition are columns in the out_auc_df
        if all(item in out_auc_df.columns for item in grp_var) and condition in out_auc_df.columns:
            pass
        else:
            print(f'{bcolors.FAIL}The grouping variable and/or the condition are not in the dataset. Please check the design file.{bcolors.ENDC}')
            # sys.exit() # This is not available in the new structure

        # create a folder within Plots named "Boxplots"
        boxplot_path = os.path.join(ROOT, OUTPUT, 'Plots', 'Boxplots')
        if not os.path.exists(boxplot_path):
            os.makedirs(boxplot_path)
        else:
            print('The Boxplot folder already exists. I will overwrite the files.')

        # clean the dataset
        out_auc_df_boxplot = out_auc_df.copy()
        out_auc_df_boxplot.dropna(subset=grp_var + [condition], inplace=True)
        # Vars to plot
        temp_vars = out_auc_df_boxplot[grp_var[0]].unique()
        # plot the y val as the f_AUC column
        y_var = [col for col in out_auc_df_boxplot.columns if col.endswith('f_AUC')][0]

        # store the inputs for the boxplot function in a zip object
        inputs = zip([out_auc_df_boxplot]*len(temp_vars),
                        [grp_var[0]]*len(temp_vars),
                        temp_vars,
                        [condition]*len(temp_vars),
                        [y_var]*len(temp_vars),
                        [os.path.join(ROOT, OUTPUT, 'Plots', 'Boxplots')+'/']*len(temp_vars))

        with get_context(start_method).Pool(n_threads) as p:
            p.starmap(plotter.plot_boxplots, tqdm(inputs, total=len(temp_vars)))


    print(f"\n{bcolors.BOLD}All analyses have been completed successfully!{bcolors.ENDC}\n")

if __name__ == '__main__':
    main()