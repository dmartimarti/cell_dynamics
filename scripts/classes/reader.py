import pandas as pd

class ExperimentReader:
    """Reads the Experiment sheet from an Excel file"""
    def __init__(self, file_path):
        self.file_path = file_path

    def read_design(self):
        # Add more methods to read other sheets in the Excel file
        xlfile = pd.read_excel(self.file_path,
                               sheet_name='Design',
                               engine='openpyxl')
        return xlfile
