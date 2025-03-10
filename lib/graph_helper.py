"""seanorn and metplotlib.pyplot helper library"""
import seaborn as sns
import matplotlib.pyplot as plt


class GraphHelper():
    """helper class for graphical representation of model, data and evaluations of model"""
    def __init__(self, cmap = 'Blues') -> None:
        self.cmap = cmap

    def show_custom_plotter(self, dataX, datay, title, xlabel, ylabel, icon, color):
        """Custom plotter with additional parameters"""
        plt.close()
        plt.figure(title)
        plt.plot(dataX, datay, icon, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def show_plotter(self, data, title, xlabel, ylabel):
        """helper function for plot graph"""
        print('show plotter')
        print(data.shape)
        plt.close()
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data)
        plt.show()
        plt.close()

    def show_heatmap_plotter(self, data, annot = True):
        """helper function to show heatmap"""
        plt.close()
        plt.figure()
        plt.plot(data)
        sns.heatmap(data.corr(), annot = annot)
        plt.show()
        plt.close()

        plt.figure()
        sns.pairplot(data)
        plt.show()
        plt.close()

    def loss_history(self, epochs, title, Xlabel, ylabel):
        """helper function to show loss of history and value during training and testing"""
        plt.close()
        plt.figure()
        plt.plot(epochs.history['loss'])
        if 'val_loss' in epochs.history:
            plt.plot(epochs.history['val_loss'])
        plt.title(title)
        plt.xlabel(Xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.close()

    def model_accuracy(self, epochs, title, Xlabel, ylabel):
        """helper function to represent model accuracy from epoch"""
        plt.close()
        plt.figure()
        plt.plot(epochs.history['accuracy'])
        plt.title(title)
        plt.xlabel(Xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.close()

    def countplot(self, data, X=None, y=None, palette=None, label=None):
        """helper function to repsetn counterplot graph"""
        plt.figure()
        sns.countplot(data = data, x =X, y=y, palette = palette, label=label)
        plt.show()
        plt.close()

    def show_heatmap(self, df, title, Xlabel, ylabel, annot = True):
        """helper function to represtn heatmap graph"""
        sns.heatmap(df, annot= annot, cmap=self.cmap)
        plt.title(title)
        plt.xlabel(Xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.close()
