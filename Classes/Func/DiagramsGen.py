import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt


class PlotMain():
    def __init__(self, save_loc=None):
        self.__safe_loc = save_loc

    def __SaveRouteGen(self, fig_n):
        if not self.__safe_loc:
            s_l = Path.cwd() / (fig_n + '.png')
        else:
            s_l = self.__safe_loc / (fig_n + '.png')

        return s_l

    def lmplot(self, x_label, y_label, df, fig_name):
        save_loc = self.__safe_loc / (fig_name + '.png')
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

    def MultiLineplot(self, x_label, y_labels, df, fig_n):
        save_loc = self.__safe_loc / (fig_n + '.png')
        sns.set_theme(style="whitegrid")
        sns.set(rc={'figure.figsize': (16, len(y_labels) * 4)})

        if type(y_labels) == dict:
            col_sel = list(y_labels.keys())
            for i in range(len(col_sel)):
                col_n = col_sel[i]
                plt.subplot(len(col_sel), 1, i + 1)
                if not y_labels[col_n]:
                    pass
                else:
                    plt.ylim(*y_labels[col_n])
                sns.lineplot(x=x_label, y=col_n,
                             data=df).set_title(col_n + '_trends')
        elif type(y_labels) == list:
            col_sel = y_labels
            for i in range(len(col_sel)):
                col_n = col_sel[i]
                # arr = np.array(df[col_n].to_list())
                # per_arr = np.diff(arr) / arr[1:] * 100
                # min_p = np.min(per_arr)
                # max_p = np.max(per_arr)
                plt.subplot(len(col_sel), 1, i + 1)
                # plt.plot(label='Per-Range: {0}% ~ {1}%'.format(min_p, max_p))
                sns.lineplot(x=x_label, y=col_n,
                             data=df).set_title(col_n + '_trends')

        plt.tight_layout()
        plt.savefig(save_loc)
        # plt.suptitle(fig_n, fontsize=12)
        plt.close()

    def lineplot(self, x_label, y_label, df, fig_name):
        #save_loc = Path(self.__safe_loc) / (fig_name + '.png')
        sns.set_theme(style='whitegrid')
        sns.set(rc={'figure.figsize': (18, 4)})
        sns.lineplot(x=x_label, y=y_label, data=df)
        plt.title(fig_name, fontsize=12)
        plt.show()
        # plt.savefig(save_loc)
        plt.close()

    def HeatMapPlot(self, df, fig_n):
        save_loc = self.__SaveRouteGen(fig_n)
        df.index = df.index.str.upper()
        sns.set(rc={'figure.figsize': (9, 8)})
        sns.heatmap(df, annot=True, linewidths=.5)
        plt.title(fig_n, fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_loc)
        plt.close()

    def SensSpecPlot(self, df, fig_n):
        save_loc = self.__SaveRouteGen(fig_n)
        sns.set_theme(style='whitegrid')
        fig_dims = (15, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        sns.lineplot(x='sep', y='sens', data=df, label='sensitivity')
        sns.lineplot(x='sep', y='spec', data=df, label='specificity')
        ax.set_xlabel('Cut Off', fontsize=14)
        ax.set_ylabel('Sens/Spec', fontsize=14)
        func = lambda x, y: [x.loc[x.sep == i].sens.item() for i in y]
        typical_cut = [75, 76.5, 100, 105]
        sens_l = func(df, typical_cut)
        plt.vlines(x=typical_cut,
                   ymin=[0, 0, 0, 0],
                   ymax=sens_l,
                   colors='teal',
                   label='vline_multiple - partial height')
        plt.title(fig_n, fontdict={'fontsize': 20})
        plt.savefig(save_loc)
        plt.close()