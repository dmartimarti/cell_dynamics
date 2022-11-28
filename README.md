# Cellular dynamics 

This script is for internal use for the Cabreiro lab. As we are heavy users of the cell reader Biospa, we are constantly analysing the output of the machine and calculating different parameters of cell growth such as AUCs or the dynamics from the timeseries. 

This script is essentially a rework from the [original script](https://github.com/PNorvaisas/Growth_analysis) used in the lab by [Pov](https://github.com/PNorvaisas), adapted to Python 3.X and with some extra features (multiprocessing, multidimensional analyses, etc). The script is still **under development** and it is not fully tested, any feedback is welcome.

## Installation

The script is written in Python 3.10 and it is recommended to use a virtual environment. The script uses the following libraries:

- numpy
- pandas
- plotly
- tqdm
- itertools
- matplotlib
- seaborn

I highly recommed to create a new conda environment and install the libraries with the following command:

```bash
conda create -n <env_name> python=3.10 numpy pandas plotly tqdm matplotlib seaborn
```

I will create a requirements.txt file in the future to make the installation easier.

## Usage

The main script is named `growth.py` and depends on the `functions.py` file. The script is called with the following command:

```bash
python growth.py -i <input_folder> -o <output_folder> -t <threads> 
```

The script can be called with the following arguments:

- `-i` or `--input`: Path to the input folder. The script will look for a file named `Design.xlsx` and read it. 
- `-o` or `--output`: Path to the output folder. The script will create a folder for plots and another for the csv files.
- `-t` or `--threads`: Number of threads to use. 

A word about the input file: for now it needs to be named exactly as `Design.xlsx` and it needs to have the following columns:
- `File`: Name of the txt files to analyse. The script will look for a file with the same name in the input folder.

If you want to include information about the plate pattern, `Design.xlsx` must have a column named `Pattern`, where it indicates the name of the Pattern file or files that it will read and parse. This pattern file can have as many sheets as you want, with a shape of a 96-well plate. If you don't want a specific column to be read by the script, you can name it starting with an underscore, e.g., `_Variable`.

### Usage example

Having the scripts in the same folder as the input `Design.xlsx` file, the following command will run the script:

```bash
python growth.py -i ./ -o Output -t 4
```

## Output

The script will create a folder named as the specified output, and within it will create a folder named `Plots`. It will save two .csv files in the output folder, one with the AUCs and another with the timeseries. Then it will save all the plots within the `Plots` folder.

## License

MIT License

Copyright (c) [2022] [Daniel Martinez Martinez]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
