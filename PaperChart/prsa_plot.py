import sys
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from pylatex.section import Paragraph
from pylatex.utils import bold, NoEscape
from pylatex import Document, Package, Figure, SubFigure, Command, Section

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import SaveGen, ConfigRead

p_name = 'PRSAPlot'
arr_set = (-2, 2, 1000)
charts_name = ['a_anchors', 'b_clips', 'c_overlap', 'd_mean']
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)


def main():
    main_p = PRSAAnalysis(arr_set)
    main_p.AnchorPlot((16, 4), (0, 120), charts_name[0])
    main_p.ClipsPlot((16, 4), charts_name[1])
    main_p.OverlapPlot((8, 4), (0, 30), charts_name[2])
    main_p.MeanPlot((8, 4), charts_name[3])
    doc = GenLatexPdf([s_f_fold / i for i in charts_name])
    doc.generate_pdf(str(s_f_fold / p_name), clean_tex=False)


class Basic():
    def __init__(self) -> None:
        pass

    def ArrGen(self, arr_st: tuple, seed_st: int = 4):
        np.random.seed(seed_st)
        arr_x = np.linspace(0, arr_st[-1], arr_st[-1], endpoint=False)
        arr_y = np.random.uniform(*arr_st)
        arr_ = np.array([arr_x, arr_y])
        return arr_

    def AnchorFunc(self, met_n: str, T: int = 1):
        if met_n == 'AC':
            anchor_set = lambda x, y: True if np.mean(x[y:y + T]) > np.mean(x[
                y - T:y]) else False
        elif met_n == 'DC':
            anchor_set = lambda x, y: True if np.mean(x[y:y + T]) < np.mean(x[
                y - T:y]) else False
        else:
            print('No match method')
            return
        return anchor_set


class PRSAAnalysis(Basic):
    def __init__(self,
                 arr_st: tuple,
                 L: int = 18,
                 T: int = 2,
                 met: str = 'AC') -> None:
        super().__init__()
        self.__L = L
        self.__arr = self.ArrGen(arr_st, seed_st=55)
        self.__met = self.AnchorFunc(met, T)
        self.__clips = self.__GetAnchor()
        self.__c_sts = ['tab:orange', 'tab:green', 'tab:purple', 'tab:blue']

    def __GetAnchor(self):
        anchor_s = []
        arr_x, arr_y = self.__arr[0], self.__arr[1]
        for i in range(self.__L, len(arr_y) - self.__L):
            anchor = {}
            if not self.__met(arr_y, i):
                anchor['ind'] = arr_x[i]
                anchor['siz'] = arr_y[i]
                anchor['val'] = False
                anchor['clip_i'] = []
                anchor['clip_v'] = []
            else:
                anchor['ind'] = arr_x[i]
                anchor['siz'] = arr_y[i]
                anchor['val'] = True
                L_clip = slice(i - self.__L, i + self.__L)
                anchor['clip_i'] = np.arange(-self.__L, self.__L, step=1)
                anchor['clip_v'] = arr_y[L_clip].tolist()
            anchor_s.append(anchor)
        return anchor_s

    def __BasicAnnotation(self,
                          ax,
                          p: tuple,
                          p_n: str,
                          c_st: str,
                          offsets: tuple = (),
                          **kwargs):
        fraction_x = (p[0] - ax.get_xlim()[0]) / (ax.get_xlim()[1] -
                                                  ax.get_xlim()[0])
        fraction_y = (p[1] - ax.get_ylim()[0]) / (ax.get_ylim()[1] -
                                                  ax.get_ylim()[0])
        ax.annotate('$' + p_n + '$',
                    xy=p,
                    xycoords='data',
                    color=c_st,
                    size=12,
                    xytext=(fraction_x + offsets[0], fraction_y + offsets[1]),
                    textcoords='axes fraction',
                    arrowprops=dict(facecolor=c_st, width=1.5, **kwargs),
                    horizontalalignment='right',
                    verticalalignment='top')
        return

    def AnchorPlot(self, dims: tuple, display_range: tuple, save_name: str):
        sns.reset_orig()
        sns.set_style('white')
        fig, ax = plt.subplots(1, 1, figsize=dims)

        # size set
        ax.set_title('$(a)$', loc='left', fontsize=15, fontstyle='italic')
        ax.set_xlim(*display_range)
        ax.set_ylim(min(self.__arr[1]) - 2, max(self.__arr[1]) + 2)
        ax.set_xlabel('$i$', fontsize=15)
        ax.set_ylabel('$x_i$', fontsize=15)

        # inital arr
        ax.scatter(*self.__arr, c='black', s=35)
        ax.plot(*self.__arr, 'k-', lw=0.8)

        # anchor arr
        ac_s = [i for i in self.__clips if i['val']]
        ac_p_s = np.array([[i['ind'] for i in ac_s], [i['siz'] for i in ac_s]])
        ax.scatter(*ac_p_s, c='firebrick', s=20)

        # anchor select
        p_n_s = ['v=1', 'v=2', 'v=3', 'v=4']
        p_offsets = [(-0.01, -0.3), (-0.03, 0.2), (-0.02, 0.4), (0.02, 0.2)]
        for i in range(len(p_n_s)):
            self.__BasicAnnotation(ax,
                                   ac_p_s.T[i],
                                   p_n_s[i],
                                   self.__c_sts[i],
                                   p_offsets[i],
                                   shrink=0.05)
        fig.tight_layout()
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()

    def ClipsPlot(self, dims: tuple, save_name: str):
        sns.reset_orig()
        sns.set_style('white')
        fig, ax_s = plt.subplots(1,
                                 4,
                                 figsize=dims,
                                 gridspec_kw={'width_ratios': [1, 1, 1, 1]})
        ac_s = [i for i in self.__clips if i['val']]
        for i in range(len(self.__c_sts)):

            # shape setting
            ax_s[i].set_xlim(ac_s[i]['clip_i'][0], ac_s[i]['clip_i'][-1])
            ax_s[i].set_ylim(min(self.__arr[1]) - 2, max(self.__arr[1]) + 2)
            ax_s[i].set_xlabel('$k$', fontsize=15)
            ax_s[i].set_ylabel('$x_{i(v)}+k$', fontsize=15)

            # scatter plot
            clip_i = ac_s[i]['clip_i']
            clip_v = ac_s[i]['clip_v']
            clip_p = np.array([clip_i, clip_v])
            ax_s[i].plot([0, 0], ax_s[i].get_ylim(), 'k-', lw=0.8)
            ax_s[i].plot(*clip_p, c=self.__c_sts[i], lw=0.8)
            ax_s[i].scatter(*clip_p, c=self.__c_sts[i], s=20)

            # text plot
            ax_s[i].text(0.5,
                         0.85,
                         '$v={0}$'.format(i + 1),
                         ha='center',
                         va='center',
                         fontsize=12,
                         fontweight='bold',
                         color=self.__c_sts[i],
                         transform=ax_s[i].transAxes,
                         bbox=dict(facecolor='white',
                                   alpha=1,
                                   edgecolor='white'))

        fig.suptitle('$(b)$', x=0.05, y=0.95, fontsize=15, fontstyle='italic')
        fig.tight_layout()
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()

    def OverlapPlot(self, dims: tuple, lap_slice: tuple, save_name: str):
        sns.reset_orig()
        sns.set_style('white')
        fig, ax = plt.subplots(1, 1, figsize=dims)
        ac_s = [i for i in self.__clips if i['val']]

        ax.set_title('$(c)$', loc='left', fontsize=15, fontstyle='italic')
        ax.set_xlim(ac_s[0]['clip_i'][0], ac_s[0]['clip_i'][-1])
        ax.set_ylim(min(self.__arr[1]) - 2, max(self.__arr[1]) + 2)
        ax.set_xlabel('$k$', fontsize=15)
        ax.set_ylabel('$x_{i(v)}+k$', fontsize=15)

        for i in range(len(ac_s))[lap_slice[0]:lap_slice[-1]]:
            clip_i = ac_s[i]['clip_i']
            clip_v = ac_s[i]['clip_v']
            clip_p = np.array([clip_i, clip_v])
            ax.plot(*clip_p, lw=0.5)
            ax.scatter(*clip_p, s=15)
        ax.plot([0, 0], ax.get_ylim(), 'k-', lw=0.8)

        ax.text(0.5,
                0.85,
                '$v={0}, ..., {1}$'.format(*lap_slice),
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold',
                color='black',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=1, edgecolor='white'))

        fig.tight_layout()
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()

    def MeanPlot(self, dims: tuple, save_name: str):
        sns.reset_orig()
        sns.set_style('white')
        fig, ax = plt.subplots(1, 1, figsize=dims)
        ac_s = [i for i in self.__clips if i['val']]

        ax.set_title('$(d)$', loc='left', fontsize=15, fontstyle='italic')
        ax.set_xlim(ac_s[0]['clip_i'][0], ac_s[0]['clip_i'][-1])
        ax.set_ylim(min(self.__arr[1]) - 2, max(self.__arr[1]) + 2)
        ax.set_xlabel('$k$', fontsize=15)
        ax.set_ylabel('$\overline{x}(k)$', fontsize=15)

        clip_i = ac_s[0]['clip_i']
        clip_v_s = np.array([i['clip_v'] for i in ac_s])
        clip_v = [np.mean(i) for i in clip_v_s.T]
        mean_p = (clip_i, clip_v)

        ax.scatter(*mean_p, c='black', s=35)
        ax.plot(*mean_p, 'k-', lw=0.8)
        ax.plot([0, 0], ax.get_ylim(), 'k-', lw=1)
        ax.plot(ax.get_xlim(), [0, 0], 'k-', lw=1)

        fig.tight_layout()
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()


def GenLatexPdf(fig_pathes: list):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    chart_description = r'''Illustration of the PRSA technique: (a) Anchor points 
    are selected from the original time series $(x_i)$; here increase events are 
    selected according to Eq. (1a), corresponding to T = 1. (b) Windows 
    (surroundings) of length 2L with L=18 are defined around each anchor point; 
    the points in each window are given by Eq. (3) and shown here for the first 
    four anchor points. (c) The surroundings of many anchor points (all located in 
    the centre) are shown on top of each other. (d) The PRSA curve $(\overline{x}(k))$ 
    resulting from averaging over all. '''

    with doc.create(Figure(position='h!')) as imagesRow1:
        doc.append(Command('centering'))
        with doc.create(SubFigure(position='c',
                                  width=NoEscape(r'1\linewidth'))) as left_row:
            left_row.add_image(str(fig_pathes[0]),
                               width=NoEscape(r'0.95\linewidth'))

    with doc.create(Figure(position='h!')) as imagesRow2:
        doc.append(Command('centering'))
        with doc.create(SubFigure(position='c',
                                  width=NoEscape(r'1\linewidth'))) as left_row:
            left_row.add_image(str(fig_pathes[1]),
                               width=NoEscape(r'0.95\linewidth'))

    with doc.create(Figure(position='h!')) as imagesRow3:
        doc.append(Command('centering'))
        with doc.create(
                SubFigure(position='c',
                          width=NoEscape(r'0.5\linewidth'))) as left_row:
            left_row.add_image(str(fig_pathes[2]),
                               width=NoEscape(r'0.95\linewidth'))

        with doc.create(
                SubFigure(position='c',
                          width=NoEscape(r'0.5\linewidth'))) as left_row:
            left_row.add_image(str(fig_pathes[3]),
                               width=NoEscape(r'0.95\linewidth'))

    with doc.create(Paragraph('')) as tail:
        tail.append(bold('Fig.2 '))
        tail.append(NoEscape(chart_description))

    return doc


if __name__ == '__main__':
    main()