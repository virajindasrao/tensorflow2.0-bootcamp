import seaborn as sns
import matplotlib.pyplot as plt


class plt_helper():
    def __init__(self, cmap = 'Blues') -> None:
        self.cmap = cmap

    def show_custom_plotter(self, dataX, datay, title, xlabel, ylabel, icon, color):
        plt.close()
        plt.figure(title)
        plt.plot(dataX, datay, icon, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def show_plotter(self, data, title, xlabel, ylabel):
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
        plt.close()
        plt.figure()
        plt.plot(epochs.history['accuracy'])
        plt.title(title)
        plt.xlabel(Xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.close()

    def countplot(self, data, X=None, y=None, palette=None, label=None):
        plt.figure()
        sns.countplot(data = data, x =X, y=y, palette = palette, label=label)
        plt.show()
        plt.close()

    def show_heatmap(self, df, title, Xlabel, ylabel, annot = True):
        sns.heatmap(df, annot= annot, cmap=self.cmap)
        plt.title(title)
        plt.xlabel(Xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.close()
        
        
