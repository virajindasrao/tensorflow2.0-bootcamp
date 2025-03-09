"""script to test binary classification case with tensorflow 2.0 sequentical model with sentiment analysis of productions example"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from lib.dataframe_helper import pd_helper
from lib.graph_helper import plt_helper
from lib.tensorflow_helper import TfModelHelper


# Rental bike class declaration
class SentimentAnalysis():
    """class for sentiment analaysis"""
    def __init__(self) -> None:
        # pandas init and basic lib setup
        self.pd = pd_helper('amazon_alexa.tsv', seperator='\t')
        # Show the data to cross verify at runtime
        self.pd.show_details()
        # Initiate helper libraries
        self.plt = plt_helper()
        self.tfh = TfModelHelper()
        #Initiate class veriables
        self.df = None
        self.X = None
        self.y = None
        self.variation_dummies = None

    def visualize_dataset(self):
        """visualize data post dataframe load"""
        df = self.pd.get_data()
        plt.figure()
        # Show feedback and rating variaations
        self.plt.countplot(df, X='feedback')
        self.plt.countplot(df, X='rating')

    def cleanup_data(self):
        """data cleanup before passing it to the model for training and evaluation"""
        self.df = self.pd.get_data()
        # Drop unnecessary data
        self.df = self.df.drop(['date','rating'], axis=1)
        # Convert variation column data values into column and values as 0 and 1
        # This will help model to train and predict the sentiments more accurately
        self.variation_dummies = pd.get_dummies(self.df['variation'], drop_first= True, dtype=int)
        # Drop variation column post row to column conversion as this will not be required for training
        # Also it may affect the model prediction capability and deviate the actual model accuracy
        self.df.drop(['variation'], axis = 1, inplace=True)
        # Merge both variation row to column dataframe into main dataframe
        self.df = pd.concat([self.df, self.variation_dummies], axis=1)
        # Initiate tokenizer
        vectorizer = CountVectorizer()
        # Tokenize the review comments
        # This process will convert all words from comments to columns
        # This will help model to predict more accurately
        df_review_vector = vectorizer.fit_transform(self.df['verified_reviews'].values.astype('U'))
        # Drop review column from the dataframe as it is not required anymore
        # also keeping it in the dataframe might affect the training and prediction
        self.df.drop(['verified_reviews'], axis = 1, inplace=True)
        # finally merge the review tokenizer dataframe with main dataframe object
        self.df = pd.concat([self.df, pd.DataFrame(df_review_vector.toarray())], axis=1)
        self.X = self.df.drop(['feedback'], axis =1)
        self.y = self.df['feedback']

    def model_build_and_train(self):
        """builds and trains model based of the configuraation"""
        self.tfh.X_train, self.tfh.X_test, self.tfh.y_train, self.tfh.y_test = train_test_split(
            self.X, self.y, test_size=0.2
            )
        # prepare a configuration to build a model
        # This will build a model with have 2 hidden layers of neuron with 400 neurons in each
        # 1 output layer with 1 neuron
        config = [
            {'units':400, 'activation':'relu', 'shape':4060},
            {'units':400, 'activation':'relu'},
            {'units':1, 'activation':'sigmoid'},
        ]
        # build and train a model based on the above configuration
        self.tfh.build_and_train(
            'sentiment_analysis',
            config,
            optimizer='Adam',
            loss='binary_crossentropy',
            metrix=['accuracy'],
            epochs=10
            )

    def evaluate_model(self):
        """evaluate model post model trained to validate the accuracy of the model"""
        self.tfh.evaluate_binary_classification_model(
            model_name='sentiment_analysis'
            )

    def visualize_model_evaluation(self):
        """visualize the model evaluation for better understanding"""
        # Show epoch loss history during training
        self.plt.loss_history(
            epochs=self.tfh.epoch_history,
            title='Model Loss Progress During Training',
            Xlabel='Epochs',
            ylabel='Training Loss'
            )
        # Show model accurary during training
        self.plt.model_accuracy(
            epochs=self.tfh.epoch_history,
            title='Model Accuracy During Training',
            Xlabel='Epochs',
            ylabel='Model Accuracy'
            )
        # Show confusion matrics of model accurary during training
        self.plt.show_heatmap(
            df=self.tfh.cm_loss,
            title='Confusion Matrics - Model Accuracy During Training',
            Xlabel='Actual Result',
            ylabel='Prediction'
            )
        # Show confusion matrics of model accurary during testing
        self.plt.show_heatmap(
            df=self.tfh.cm_accuracy,
            title='Confusion Matrics - Model Accuracy During Testing',
            Xlabel='Actual Result',
            ylabel='Prediction'
            )

if __name__ == "__main__":
    # Load helper class and initiate dataframe
    sa = SentimentAnalysis()
    # Print dataframe details
    sa.pd.show_details()
    # Represent dataframe details in graph view
    sa.visualize_dataset()
    # Cleanup data before training a model
    sa.cleanup_data()
    # Build and train a model
    sa.model_build_and_train()
    # Evaluate model post training
    sa.evaluate_model()
    # Visualize model evaluation in graphical representation for better understanding
    sa.visualize_model_evaluation()
