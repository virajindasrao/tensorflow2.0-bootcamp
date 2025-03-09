"""tensorflow helper library"""
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix



class TfModelHelper():
    """This class is a helper library of terraform furnctions"""
    def __init__(self) -> None:
        self.model, self.epoch_history = None, None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.scaler, self.y_predict = None, None
        self.y_predict_orig, self.y_test_orig = None, None
        self.cm_loss , self.cm_accuracy = None, None

    def build_and_train(
            self,
            model_name,
            config,
            optimizer='Adam',
            loss = 'mean_squared_error',
            metrix = None,
            epochs=None,
            batch_size=None,
            validation_split=None
            ):
        """this function builds and trains tensorflow model using given parameters 
        like optimizer, loss, metrics etc"""
        self.model = tf.keras.models.Sequential()

        self.model.add(
            tf.keras.layers.Dense(
                units = config[0]['units'],
                activation = config[0]['activation'],
                input_shape=(config[0]['shape'],)
                )
            )
        for k in config[1:]:
            self.model.add(tf.keras.layers.Dense(units = k['units'], activation = k['activation']))

        print(self.model.summary())
        print(f'get_weights : {self.model.get_weights()}')
        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrix)
        self.epoch_history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs = epochs, 
            batch_size = batch_size, 
            validation_split = validation_split)
        self.model.save(model_name+'.keras')

    def load_model(self, model_name):
        """loads the saved model of given name as parameter to the function.
        Panic error if model is not found"""
        self.model = tf.keras.models.load_model(model_name+'.keras')


    def evaluate_regression_model(self, model_name):
        """This function evaluated the regresion model"""
        self.load_model(model_name)
        self.y_predict = self.model.predict(self.X_test)
        self.y_predict_orig = self.scaler.inverse_transform(self.y_predict)
        self.y_test_orig = self.scaler.inverse_transform(self.y_test)
        k = self.X_test.shape[1]
        n = len(self.X_test)
        rsme = float(
            format(
                np.sqrt(
                    mean_squared_error(
                        self.y_test_orig,
                        self.y_predict_orig
                        )
                    ),
                '0.3f')
            )
        mse = mean_squared_error(self.y_test_orig, self.y_predict_orig)
        mae = mean_absolute_error(self.y_test_orig, self.y_predict_orig)
        r2 = r2_score(self.y_test_orig, self.y_predict_orig)
        adj_r2 = 1 - (1 - r2)*(n-1)/(n-k-1)
        print(f' rsme : {rsme}')
        print(f' mse : {mse}')
        print(f' mae : {mae}')
        print(f' r2 : {r2}')
        print(f' adj_r2 : {adj_r2}')

    def evaluate_binary_classification_model(self, model_name):
        """this functio evaluates the training of the binary classic tensorflow model"""
        self.load_model(model_name)
        y_pred_train = self.model.predict(self.X_train)
        y_pred_train = y_pred_train > 0.5
        self.cm_loss = confusion_matrix(self.y_train, y_pred_train)

        y_pred_test = self.model.predict(self.X_test)
        y_pred_test = y_pred_test > 0.5
        self.cm_accuracy = confusion_matrix(self.y_test, y_pred_test)
