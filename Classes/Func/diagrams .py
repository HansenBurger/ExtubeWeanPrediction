import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt


class PlotMain():
    def __init__(self, save_loc=str(Path.cwd())):
        self.__safe_loc = save_loc

    def lmplot(self, x_label, y_label, df, fig_name):
        save_loc = Path(self.__safe_loc) / (fig_name + '.png')
        sns.set_theme(style='whitegrid')
        sns.lmplot(x=x_label,
                   y=y_label,
                   data=df,
                   fit_reg=False,
                   height=3.3,
                   aspect=5.5)
        plt.title(fig_name, fontsize=12)
        plt.show()
        plt.close()

    def MuiltLmplot(self, x_label):
        pass

    def lineplot(self, x_label, y_label, df, fig_name):
        save_loc = Path(self.__safe_loc) / (fig_name + '.png')
        sns.set_theme(style='whitegrid')
        sns.set(rc={'figure.figsize': (18, 7)})
        sns.lineplot(x=x_label, y=y_label, data=df)
        plt.title(fig_name, fontsize=12)
        plt.savefig(save_loc)
        plt.close()