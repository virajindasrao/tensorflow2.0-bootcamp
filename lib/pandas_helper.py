import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


class pd_helper():
    def __init__(self, file_name) -> None:
        self.file = r'C:\Users\viraj\Projects\tensorflow2.0-bootcamp\data/'+file_name
        self.obj = pd.read_csv(self.file)

    def get_data(self):
        return self.obj
            
    def set_data(self, data):
        self.obj = data

    def show_details(self):
        head = self.obj.head(5)
        print(head)

        desc = self.obj.describe()
        print(desc)

        info = self.obj.info()
        print(info)

    
    def get_data_as_frequent(self, column, frequency, linewidth):
        return self.obj[column].asfreq(frequency).plot(linewidth = linewidth)

    def get_data_by_column(self, columns):
        return self.obj[columns]
    
    def drop_column(self, column, axis = 1):
        self.obj = self.obj.drop(labels = [column], axis = axis)

    def change_datatime_format(self, data, col_data):
        col_data = self.obj.to_datetime(col_data, format = r'%m/%d/%Y')
        return col_data
    
    def set_index(self, index_col, type='datetimeindex'):
        if type == 'datetimeindex':
            self.obj.index = pd.DatetimeIndex(index_col)

        