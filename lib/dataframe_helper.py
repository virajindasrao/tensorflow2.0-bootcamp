"""helper library for dataframe operation"""
import os
import pandas as pd


class pd_helper():
    """helper class for dataframe operations"""
    def __init__(self, file_name, seperator=None) -> None:
        relative_path = os.path.join("data", file_name)  # Refers to "data/file.txt" in the current directory
        absolute_path = os.path.abspath(relative_path)    # Converts it to an absolute path
        self.file = absolute_path
        self.obj = pd.read_csv(self.file, sep=seperator)

    def get_data(self):
        """get dataframe"""
        return self.obj

    def set_data(self, data):
        """set dataframe"""
        self.obj = data

    def show_details(self):
        """show dataframe details"""
        head = self.obj.head(5)
        print(head)

        desc = self.obj.describe()
        print(desc)

        info = self.obj.info()
        print(info)

    def get_data_as_frequent(self, column, frequency, linewidth):
        """get data as frequent function for data fetching based on the frequency"""
        return self.obj[column].asfreq(frequency).plot(linewidth = linewidth)

    def get_data_by_column(self, columns):
        """Get data based on the column"""
        return self.obj[columns]

    def drop_column(self, column, axis = 1):
        """drop a column from dataframe"""
        self.obj = self.obj.drop(labels = [column], axis = axis)

    def change_datatime_format(self, col_data):
        """change the format of date type column in dataframe"""
        col_data = self.obj.to_datetime(col_data, format = r'%m/%d/%Y')
        return col_data
    
    def set_index(self, index_col, type='datetimeindex'):
        """change index of datafraome"""
        if type == 'datetimeindex':
            self.obj.index = pd.DatetimeIndex(index_col)

        