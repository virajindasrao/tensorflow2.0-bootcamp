from lib.pandas_helper import pd_helper
from lib.matplotlib_helper import plt_helper
from lib.tensorflow_helper import tf_helper

# Rental bike class declaration
class rental_bike():
    def __init__(self) -> None:
        # pandas init and basic lib setup
        self.pd = pd_helper('bike_sharing_daily.csv')
        # Show the data to cross verify at runtime
        self.pd.show_details()
        self.plt = plt_helper()
        self.tf_helper = tf_helper()
        self.x_numerical = None
        self.x_cat = None

    def cleanup_dataset(self):
        self.pd.drop_column('instant')
        self.pd.drop_column('registered')
        self.pd.drop_column('casual')

        data = self.pd.get_data()
        # data.dteday = self.pd.change_datatime_format(data, data.dteday)
        self.pd.set_data(data)
        self.pd.set_index(data.dteday, 'datetimeindex')
        data = self.pd.get_data()
        self.pd.drop_column('dteday')
        
    def visualize_dataset(self):
        self.pd.get_data_as_frequent('cnt', 'W', 3)
        self.plt.show_plotter(self.pd.get_data(), 'Bike usage per week', 'Week', 'Bike Rental')
        self.x_numerical = self.pd.get_data_by_column(['temp','hum','windspeed','cnt'])
        self.plt.show_heatmap_plotter(self.x_numerical)

    def create_training_testing_data(self):
        self.x_cat = self.pd.get_data_by_column(['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'])
        self.tf_helper.create_training_testing_data(self.x_numerical, self.x_cat)


if __name__ == "__main__":
    bike = rental_bike()
    data = bike.cleanup_dataset()
    x_numerical = bike.visualize_dataset()
    bike.create_training_testing_data()
    config = [
        {'units':100, 'activation':'relu', 'shape':35},
        {'units':100, 'activation':'relu'},
        {'units':100, 'activation':'relu'},
        {'units':1, 'activation':'linear'},
    ]

    bike.tf_helper.build_and_train('bike-rental', config)
    bike.plt.loss_history(bike.tf_helper.epoch_history, 'Model loss  progress during training', 'epochs', 'Training Loss')
    bike.tf_helper.evaluate_model('bike-rental', 'Model predictions', 'True values')

