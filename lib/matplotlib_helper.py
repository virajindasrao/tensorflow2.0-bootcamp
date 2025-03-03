import seaborn as sns
import matplotlib.pyplot as plt


class plt_helper():
    def __init__(self) -> None:
        pass
    
    def show_plotter(self, data, title, xlabel, ylabel):
        plt.close()
        print(f'show plotter {data}, {title}, {xlabel}, {ylabel}')
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(data)
        plt.show()
        plt.savefig('plotter.pdf')

    def show_heatmap_plotter(self, data, annot = True):
        print(f'show_heatmap_plotter {data}')
        plt.close()
        plt.figure()
        plt.plot(data)
        sns.heatmap(data.corr(), annot = annot)
        plt.show()
        plt.savefig('show_heatmap_plotter.pdf')
        plt.close()

        plt.figure()
        sns.pairplot(data)
        sns.heatmap(data.corr(), annot = annot)
        plt.show()
        plt.savefig('show_heatmap_plotter2.pdf')

    def loss_history(self, epochs, title, xlabel, ylabel):
        plt.close()
        plt.figure()
        plt.plot(epochs.history['loss'])
        if 'val_loss' in epochs.history:
            plt.plot(epochs.history['val_loss'])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.savefig('loss_history.pdf')
