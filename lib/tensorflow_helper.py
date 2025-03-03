import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt


class tf_helper():
    def __init__(self) -> None:
        self.model = None
        self.epoch_history = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaler = None

    def build_and_train(self, model_name, config):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Dense(units = config[0]['units'], activation = config[0]['activation'], input_shape=(config[0]['shape'],)))
        print(f'config[1:-1] {config[1:-1]}')
        for k in config[1:]:
            self.model.add(tf.keras.layers.Dense(units = k['units'], activation = k['activation']))

        print(self.model.summary())
        self.model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
        print(f'self.X_train {self.X_train}')
        self.epoch_history = self.model.fit(self.X_train, self.y_train, epochs = 190, batch_size = 50, validation_split = 0.2)
        self.model.save(model_name+'.keras')


    def create_training_testing_data(self, x_numerical, x_cat):
        oneHotEncoder = OneHotEncoder()
        x_cat = oneHotEncoder.fit_transform(x_cat).toarray()
        x_cat = pd.DataFrame(x_cat)
        x_numerical = x_numerical.reset_index()
        x_all = pd.concat([x_cat, x_numerical], axis = 1)
        x_all = x_all.drop('dteday', axis = 1)
        print(x_all)
        x = x_all.iloc[:, :-1].values
        y = x_all.iloc[:,-1:].values
        print(x.shape)
        print(y.shape)
        self.scaler = MinMaxScaler()
        y = self.scaler.fit_transform(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)

    def evaluate_model(self, model_name, x_label, y_label):
        model = tf.keras.models.load_model(model_name+'.keras')
        y_predict = model.predict(self.X_test)
        plt.close()
        plt.figure('evaluate_model')
        plt.plot(self.y_test, y_predict, '^', color='blue')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        plt.savefig('evaluate_model.pdf')

        y_predict_orig = self.scaler.inverse_transform(y_predict)
        y_test_orig = self.scaler.inverse_transform(self.y_test)
        plt.plot(y_test_orig, y_predict_orig, '^', color='blue')
        plt.xlabel('Model prediction')
        plt.ylabel('True values')
        plt.show()

        k = self.X_test.shape[1]
        n = len(self.X_test)
        rsme = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)), '0.3f'))
        mse = mean_squared_error(y_test_orig, y_predict_orig)
        mae = mean_absolute_error(y_test_orig, y_predict_orig)
        r2 = r2_score(y_test_orig, y_predict_orig)
        adj_r2 = 1 - (1 - r2)*(n-1)/(n-k-1)
        print(f' rsme : {rsme}')
        print(f' mse : {mse}')
        print(f' mae : {mae}')
        print(f' r2 : {r2}')
        print(f' adj_r2 : {adj_r2}')
