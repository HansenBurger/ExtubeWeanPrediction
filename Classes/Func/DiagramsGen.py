import pandas as pd
import seaborn as sns
from numpy import ndarray
from pathlib import Path, PurePath
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


class PlotMain():
    def __init__(self, save_loc: Path = Path.cwd()):
        self.__safe_loc = Path(save_loc) if not isinstance(
            save_loc, PurePath) else save_loc

    def __SaveRouteGen(self, fig_n: str) -> Path:
        s_l = self.__safe_loc / (fig_n + '.png')
        return s_l

    def lmplot(self, x_label, y_labels, df, fig_name):
        #s ave_loc = self.__safe_loc / (fig_name + '.png')
        save_loc = self.__SaveRouteGen(fig_name)
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(1 * df.shape[0], 6))
        plt.plot([0, df.shape[0]], [0.7, 0.7], 'k--')
        plt.ylim([0.0, 1.0])
        plt.xlim([1, df.shape[0]])
        color = ['r', 'g', 'b', 'c', 'm']
        for i in range(len(y_labels)):
            plt.plot(df[x_label].tolist(),
                     df[y_labels[i]].tolist(),
                     color[i],
                     label=y_labels[i].split('_')[0])
        plt.legend(loc='best')
        plt.title(fig_name, fontsize=12)
        plt.tight_layout()
        plt.savefig(save_loc)
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
        # save_loc = Path(self.__safe_loc) / (fig_name + '.png')
        sns.set_theme(style='whitegrid')
        # sns.set(rc={'figure.figsize': (18, 4)})
        plt.figure(figsize=(18, 4))
        sns.lineplot(x=x_label, y=y_label, data=df, markers=True)
        plt.title(fig_name, fontsize=12)
        plt.show()
        # plt.savefig(save_loc)
        plt.close()

    def RocMultiPlot(self, x_labels: list, y_label: str, df: pd.DataFrame,
                     fig_n: str) -> None:
        save_loc = self.__SaveRouteGen(fig_n)
        x_labels = [x_labels] if type(x_labels) is not list else x_labels
        fig_dims = (6, 6)
        plt.subplots(figsize=fig_dims)
        plt.title('Receiver operating characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        for x_label in x_labels:
            roc_auc = round(roc_auc_score(df[y_label], df[x_label]), 2)
            fpr, tpr, _ = roc_curve(df[y_label], df[x_label])
            plt.plot(fpr,
                     tpr,
                     label='ROC: {0} (AUC = {1})'.format(x_label, roc_auc))
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_loc)
        plt.close()

    def RocSinglePlot(self,
                      true_l: ndarray,
                      pred_l: ndarray,
                      fig_n: str,
                      fig_dims: tuple = (6, 6)) -> None:
        save_loc = self.__SaveRouteGen(fig_n)
        fpr, tpr, _ = roc_curve(true_l, pred_l)
        auc = round(roc_auc_score(true_l, pred_l), 3)
        plt.subplots(figsize=fig_dims)
        plt.title('Receiver operating characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot(fpr, tpr, label='ROC AUC = {0}'.format(auc))
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_loc)
        plt.close()

    def HeatMapPlot(self, df: any, fig_n: str, cmap: str = "rocket"):
        save_loc = self.__SaveRouteGen(fig_n)
        try:
            df.index = df.index.str.upper()
        except:
            pass
        fig_wid = df.shape[1]
        fig_het = df.shape[0] - 2
        sns.set(rc={'figure.figsize': (fig_wid, fig_het)})
        # sns.heatmap(df, annot=True, linewidths=.5)
        sns.heatmap(df, annot=True, linewidths=.5, cmap=cmap)
        plt.title(fig_n, fontsize=18, fontweight='bold')
        # plt.ylabel('Scale S')
        # plt.xlabel('Anchor Len T')
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

    def ViolinPlot(self,
    x:str,
                   y: str,
                   df: pd.DataFrame,
                   fig_n: str,
                   hue: str = None):
        save_loc = self.__SaveRouteGen(fig_n)
        fig_dims = (5, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        sns.set_theme(style="whitegrid")
        sns.violinplot(x=x,y=y, data=df, hue=hue)
        plt.tight_layout()
        plt.savefig(save_loc)
        plt.close()
