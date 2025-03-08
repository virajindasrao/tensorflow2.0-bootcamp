from lib.dataframe_helper import pd_helper
from lib.graph_helper import plt_helper
from lib.tensorflow_helper import TfModelHelper
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# Rental bike class declaration
class sentiment_analysis():
    def __init__(self) -> None:
        # pandas init and basic lib setup
        self.pd = pd_helper('amazon_alexa.tsv', seperator='\t')
        # Show the data to cross verify at runtime
        self.pd.show_details()
        self.plt = plt_helper()
        self.tfh = TfModelHelper()

    def visualize_dataset(self):
        df = self.pd.get_data()
        plt.figure()
        self.plt.countplot(df, X='feedback')
        self.plt.countplot(df, X='rating')

    def cleanup_data(self):
        self.df = self.pd.get_data()
        self.df = self.df.drop(['date','rating'], axis=1)
        self.variation_dummies = pd.get_dummies(self.df['variation'], drop_first= True, dtype=int)
        self.df.drop(['variation'], axis = 1, inplace=True)
        self.df = pd.concat([self.df, self.variation_dummies], axis=1)
        vectorizer = CountVectorizer()
        df_review_vector = vectorizer.fit_transform(self.df['verified_reviews'].values.astype('U'))
        self.df.drop(['verified_reviews'], axis = 1, inplace=True)
        self.df = pd.concat([self.df, pd.DataFrame(df_review_vector.toarray())], axis=1)
        self.X = self.df.drop(['feedback'], axis =1)
        self.y = self.df['feedback']

    def model_build_and_train(self):
        self.tfh.X_train, self.tfh.X_test, self.tfh.y_train, self.tfh.y_test = train_test_split(
            self.X, self.y, test_size=0.2
            )

        config = [
            {'units':400, 'activation':'relu', 'shape':4060},
            {'units':400, 'activation':'relu'},
            {'units':1, 'activation':'sigmoid'},
        ]

        self.tfh.build_and_train(
            'sentiment_analysis',
            config,
            optimizer='Adam',
            loss='binary_crossentropy',
            metrix=['accuracy'],
            epochs=10
            )

    def evaluate_model(self):
        self.tfh.evaluate_binary_classification_model(
            model_name='sentiment_analysis'
            )

    def visualize_model_evaluation(self):
        self.plt.loss_history(
            epochs=sa.tfh.epoch_history,
            title='Model Loss Progress During Training',
            Xlabel='Epochs',
            ylabel='Training Loss'
            )
        self.plt.model_accuracy(
            epochs=sa.tfh.epoch_history,
            title='Model Accuracy During Training',
            Xlabel='Epochs',
            ylabel='Model Accuracy'
            )

        self.plt.show_heatmap(
            df=sa.tfh.cm_loss,
            title='Confusion Matrics - Model Accuracy During Training',
            Xlabel='Actual Result',
            ylabel='Prediction'
            )

        self.plt.show_heatmap(
            df=sa.tfh.cm_accuracy,
            title='Confusion Matrics - Model Accuracy During Testing',
            Xlabel='Actual Result',
            ylabel='Prediction'
            )

if __name__ == "__main__":
    sa = sentiment_analysis()
    sa.pd.show_details()
    sa.visualize_dataset()
    sa.cleanup_data()
    sa.model_build_and_train()
    sa.evaluate_model()
    sa.visualize_model_evaluation()
