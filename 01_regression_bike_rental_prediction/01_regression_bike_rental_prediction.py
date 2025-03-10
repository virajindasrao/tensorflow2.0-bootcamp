"""Script to test regression usecase using tensorflow 2.0 sequential model
to predict the affect of weather and weekdays on bike renting"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from lib.dataframe_helper import DataframHelper
from lib.graph_helper import GraphHelper
from lib.tensorflow_helper import TfModelHelper


# Rental bike class declaration
class rental_bike():
    """Class helper for regression usecase with bike rental prediction"""
    def __init__(self) -> None:
        # pandas init and basic lib setup
        self.pd = DataframHelper('bike_sharing_daily.csv')
        # Show the data to cross verify at runtime
        self.pd.show_details()
        # Initiate helper libraries
        self.graph = GraphHelper()
        self.tf_helper = TfModelHelper()
        # Initiate class veriables
        self.x_numerical = None
        self.x_cat = None

    def cleanup_dataset(self):
        """data cleanup before model building and training"""
        # Drop columns which are not required
        # and might affect the model accuracy
        self.pd.drop_column('instant')
        self.pd.drop_column('registered')
        self.pd.drop_column('casual')

        data = self.pd.get_data()
        self.pd.set_data(data)
        self.pd.set_index(data.dteday, 'datetimeindex')
        data = self.pd.get_data()
        self.pd.drop_column('dteday')

    def visualize_dataset(self):
        """Function to show the dataframe details post loading"""
        self.pd.get_data_as_frequent('cnt', 'W', 3)
        self.graph.show_plotter(data=self.pd.get_data(
        ), title='Bike usage per week', xlabel='Week', ylabel='Bike Rental')
        self.x_numerical = self.pd.get_data_by_column(
            ['temp', 'hum', 'windspeed', 'cnt'])
        self.graph.show_heatmap_plotter(self.x_numerical)

    def create_training_testing_data(self):
        """Create a testing dataframe"""
        self.x_cat = self.pd.get_data_by_column(
            ['season', 'yr', 'mnth', 'holiday',
                'weekday', 'workingday', 'weathersit']
        )
        oneHotEncoder = OneHotEncoder()
        self.x_cat = oneHotEncoder.fit_transform(self.x_cat).toarray()
        self.x_cat = pd.DataFrame(self.x_cat)
        self.x_numerical = self.x_numerical.reset_index()
        x_all = pd.concat([self.x_cat, self.x_numerical], axis=1)
        x_all = x_all.drop('dteday', axis=1)
        x = x_all.iloc[:, :-1].values
        y = x_all.iloc[:, -1:].values
        self.tf_helper.scaler = MinMaxScaler()
        y = self.tf_helper.scaler.fit_transform(y)
        self.tf_helper.X_train, self.tf_helper.X_test, self.tf_helper.y_train, self.tf_helper.y_test = train_test_split(x, y, test_size=0.2)

    def visualize_model_predictoin(self):
        '''visualizes the model prediction using plt graph library'''
        self.graph.show_custom_plotter(
            self.tf_helper.y_test_orig,
            self.tf_helper.y_predict_orig,
            title='Model prediction',
            icon='^',
            xlabel='Model prediction',
            ylabel='True values',
            color='blue'
        )


if __name__ == "__main__":
    # Initiate class object and loan dataframe
    bike = rental_bike()
    # Cleanup data before build training a model
    bike.cleanup_dataset()
    # Visualize the dataframe details
    bike.visualize_dataset()
    # Create training and testing data
    bike.create_training_testing_data()
    # prepare a configuration to build a model
    # This will build a model with have 2 hidden layers of neuron with 100 neurons in each
    # 1 output layer with 1 neuron
    config = [
        {'units': 100, 'activation': 'relu', 'shape': 35},
        {'units': 100, 'activation': 'relu'},
        {'units': 1, 'activation': 'linear'},
    ]
    # Build and train model with above configuration
    bike.tf_helper.build_and_train('bike-rental', config, epochs=150)
    # Showcase loss history
    bike.plt.loss_history(bike.tf_helper.epoch_history,
                          'Model loss  progress during training', 'epochs', 'Training Loss')
    # Evaludate model predictions using testing data
    bike.tf_helper.evaluate_regression_model('bike-rental')
    # Visualize model prediction accuracy
    bike.visualize_model_predictoin()
