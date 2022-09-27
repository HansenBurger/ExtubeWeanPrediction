from re import S
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import matplotlib.transforms as transforms

from pylatex.section import Paragraph
from pylatex.utils import bold, NoEscape
from pylatex import Document, Package, Figure, SubFigure, Command, Section

sys.path.append(str(Path.cwd()))

from Classes.Func.KitTools import SaveGen, ConfigRead

p_name = 'PoincarePlot'
resp_dict = {'type': 'v_t_i', 'name': 'V_T', 'unit': 'mL'}
single_dims = (8, 8)
lim_size = (3, 0.8)
json_loc = Path.cwd() / 'PaperChart' / '_source.json'
s_f_fold = SaveGen(ConfigRead('ResultSave', 'Form'), p_name)
chart_names = ['SD', 'PI', 'GI', 'SI', 'Total']


def main():
    p_main = PoincarePlot(resp_dict, single_dims)
    p_main.SD(chart_names[0])
    p_main.PI(chart_names[1])
    p_main.GI(chart_names[2])
    p_main.SI(chart_names[3])
    p_main.CombinePlot(chart_names[4])
    doc = GenLatexPdf([s_f_fold / (i + '.png') for i in chart_names])
    doc.generate_pdf(str(s_f_fold / p_name), clean_tex=False)


class Basic():
    def __init__(self) -> None:
        pass

    def LoadData(self, ind_type: str) -> tuple:
        df_path = ConfigRead('PoincarePlot', 'source', json_loc)
        resp_df = pd.read_csv(df_path)
        col_names = [i for i in resp_df.columns if ind_type in i]
        ind_values = resp_df[col_names].values
        ind_v_0 = np.array([i[0] for i in ind_values])
        ind_v_1 = np.array([i[1] for i in ind_values])
        return ind_v_0, ind_v_1

    def LabelName(self, ind_name: str, ind_unit) -> tuple:
        if len(ind_name.split('_')) > 1:
            n_c, n_l = ind_name.split('_')
            x_name = n_c + '_' + '{' + n_l + '(i)' + '}'
            y_name = n_c + '_' + '{' + n_l + '(i+1)' + '}'
        else:
            x_name = n_c + '_' + '{' + '(i)' + '}'
            y_name = n_c + '_' + '{' + '(i+1)' + '}'
        xlabel_ = x_name + '(' + ind_unit + ')'
        ylabel_ = y_name + '(' + ind_unit + ')'
        return xlabel_, ylabel_


class PoincarePlot(Basic):
    def __init__(self, ind_info: dict, fig_dim: tuple) -> None:
        super().__init__()
        self.__dims = fig_dim
        self.__ind = ind_info['name']
        self.__arr_0, self.__arr_1 = self.LoadData(ind_info['type'])
        self.__x_l, self.__y_l = self.LabelName(ind_info['name'],
                                                ind_info['unit'])

    def __OffAxisDistance(self, ax) -> tuple:
        dists = {}
        for i in range(len(self.__arr_0)):
            if not ax.get_xlim()[0] <= self.__arr_0[i] <= ax.get_xlim()[1]:
                continue
            elif not ax.get_ylim()[0] <= self.__arr_1[i] <= ax.get_ylim()[1]:
                continue
            else:
                p_ = (self.__arr_0[i], self.__arr_1[i])
                l_ = euclidean_distance(1, 0, p_)
                l_ = l_ * (1 if p_[0] < p_[1] else -1)
                dists[i] = l_
        dists_ac = dict(sorted(dists.items(), key=lambda item: item[1]))
        dists_dc = dict(
            sorted(dists.items(), key=lambda item: item[1], reverse=True))
        return dists_ac, dists_dc

    def __OffAxisAngle(self, ax) -> tuple:
        angles = {}
        theta_count = lambda x, y: np.degrees(np.arctan(y / x)) - 45
        for i in range(len(self.__arr_0)):
            if not ax.get_xlim()[0] <= self.__arr_0[i] <= ax.get_xlim()[1]:
                continue
            elif not ax.get_ylim()[0] <= self.__arr_1[i] <= ax.get_ylim()[1]:
                continue
            else:
                angles[i] = theta_count(self.__arr_0[i], self.__arr_1[i])
        angles_ac = dict(sorted(angles.items(), key=lambda item: item[1]))
        angles_dc = dict(
            sorted(angles.items(), key=lambda item: item[1], reverse=True))
        return angles_ac, angles_dc

    def __ScatterGen(self, ax: any) -> None:
        sns.reset_orig()
        sns.set_theme(style='white')
        ax.scatter(self.__arr_0, self.__arr_1, c='black', s=65)
        ax.scatter(self.__arr_0, self.__arr_1, c='gold', s=40)
        ax.set_xlim(
            min(self.__arr_0) * lim_size[0],
            max(self.__arr_0) * lim_size[1])
        ax.set_ylim(
            min(self.__arr_1) * lim_size[0],
            max(self.__arr_1) * lim_size[1])
        ax.set_xlabel('$' + self.__x_l + '$', fontsize=20)
        ax.set_ylabel('$' + self.__y_l + '$', fontsize=20)
        return

    def __SDPlot(self, ax: any) -> None:
        self.__ScatterGen(ax)
        ellipse_, wid, het = confidence_ellipse(self.__arr_0,
                                                self.__arr_1,
                                                ax,
                                                n_std=1.5,
                                                edgecolor='black',
                                                lw=2.5)
        p_c, l_a, s_a = centroid_connection(self.__arr_0, self.__arr_1)
        sd_1, sd_2 = points_sd(wid, het, p_c, l_a, s_a)
        ax.add_patch(ellipse_)
        ax.plot(*l_a, 'k--', linewidth=2)
        ax.plot(*s_a, 'k--', linewidth=2)
        ax.plot(*np.array([p_c, sd_1]).T, 'r-', linewidth=5)
        ax.plot(*np.array([p_c, sd_2]).T, 'g-', linewidth=5)
        ax.annotate('SD1',
                    xy=sd_1,
                    xycoords='data',
                    color='firebrick',
                    fontweight='bold',
                    size=30,
                    xytext=(sd_1[0] / max(self.__arr_0),
                            sd_1[1] / max(self.__arr_1) - 0.3),
                    textcoords='axes fraction',
                    arrowprops=dict(facecolor='firebrick', shrink=0.05),
                    horizontalalignment='right',
                    verticalalignment='top')
        ax.annotate('SD2',
                    xy=sd_2,
                    xycoords='data',
                    color='forestgreen',
                    fontweight='bold',
                    size=30,
                    xytext=(sd_2[0] / max(self.__arr_0) + 0.3,
                            sd_2[1] / max(self.__arr_1)),
                    textcoords='axes fraction',
                    arrowprops=dict(facecolor='forestgreen', shrink=0.05),
                    horizontalalignment='right',
                    verticalalignment='top')

    def SD(self, save_name: str):
        fig, ax = plt.subplots(1, 1, figsize=self.__dims)

        self.__SDPlot(ax)
        fig.tight_layout()
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()

    def __LIAnnotation(self, ax):
        ax.plot(ax.get_xlim(),
                ax.get_ylim(),
                c='darkgreen',
                linestyle='-',
                linewidth=2.5)
        p_c, _, _ = centroid_connection(self.__arr_0, self.__arr_1)
        ax.annotate('LI',
                    xy=p_c,
                    xycoords='data',
                    color='black',
                    fontweight='bold',
                    size=30,
                    xytext=(p_c[0] / max(self.__arr_0) + 0.1,
                            p_c[1] / max(self.__arr_1) - 0.3),
                    textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', width=2),
                    horizontalalignment='right',
                    verticalalignment='top')
        li_ = np.array([ax.get_xlim(), ax.get_ylim()]).T
        li_0, li_1 = li_[0], li_[1]
        return li_0, li_1

    def __BasicAnnotation(self,
                          ax,
                          p: tuple,
                          p_n: str,
                          offsets: tuple = (),
                          **kwargs):
        ax.annotate('$' + p_n + '$',
                    xy=p,
                    xycoords='data',
                    color='black',
                    size=25,
                    xytext=(p[0] / max(self.__arr_0) + offsets[0],
                            p[1] / max(self.__arr_1) + offsets[1]),
                    textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', width=2, **kwargs),
                    horizontalalignment='right',
                    verticalalignment='top')
        return

    def __PIPlot(self, ax: any) -> None:
        self.__ScatterGen(ax)
        self.__LIAnnotation(ax)
        d_ac, d_dc = self.__OffAxisDistance(ax)
        # point: RR=0, lowest
        p_0_s = sorted(
            [self.__arr_0[k] for k, v in d_ac.items() if np.abs(v) < 5])
        p_0 = [p_0_s[0]] * 2
        self.__BasicAnnotation(ax,
                               p=p_0,
                               p_n='\Delta {0}=0'.format(self.__ind),
                               offsets=(-0.1, 0.1))
        # point: RR>0 , random
        # p_1_i = random.choice(list(d_dc.keys())[0:10])
        p_1_i = list(d_dc.keys())[3]
        p_1 = (self.__arr_0[p_1_i], self.__arr_1[p_1_i])
        self.__BasicAnnotation(ax,
                               p=p_1,
                               p_n='\Delta {0}>0'.format(self.__ind),
                               offsets=(-0.05, 0.1))
        # point: RR<0, random
        # p_2_i = random.choice(list(d_ac.keys())[0:10])
        p_2_i = list(d_ac.keys())[3]
        p_2 = (self.__arr_0[p_2_i], self.__arr_1[p_2_i])
        self.__BasicAnnotation(ax,
                               p=p_2,
                               p_n='\Delta {0}<0'.format(self.__ind),
                               offsets=(0.2, -0.1))

    def PI(self, save_name: str):
        fig, ax = plt.subplots(1, 1, figsize=self.__dims)

        self.__PIPlot(ax)
        fig.tight_layout()
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()

    def __GIPlot(self, ax: any) -> None:
        self.__ScatterGen(ax)
        li_0, li_1 = self.__LIAnnotation(ax)
        d_ac, d_dc = self.__OffAxisDistance(ax)
        # point: P_i(RR_i < RR_{i+1})
        p_i_i = list(d_dc.keys())[0]
        p_i = (self.__arr_0[p_i_i], self.__arr_1[p_i_i])
        p_i_hf = plumbline_set(li_0, li_1, p_i)
        l_i_hf = np.array([p_i, p_i_hf]).T
        p_i_m = tuple((p_i[i] + p_i_hf[i]) / 2 for i in range(2))
        ax.plot(*l_i_hf, color='black', linestyle='-', linewidth=1.2)
        ax.text(*p_i, '$P_i$', fontsize=30)
        self.__BasicAnnotation(ax,
                               p=p_i_m,
                               p_n='D_i^+',
                               offsets=(0.1, 0.2),
                               shrink=0.05)

        # point: P_j(RR_i > RR_{i+1})
        p_j_i = list(d_ac.keys())[0]
        p_j = (self.__arr_0[p_j_i], self.__arr_1[p_j_i])
        p_j_hf = plumbline_set(li_0, li_1, p_j)
        l_j_hf = np.array([p_j, p_j_hf]).T
        p_j_m = tuple((p_j[i] + p_j_hf[i]) / 2 for i in range(2))
        ax.plot(*l_j_hf, color='black', linestyle='-', linewidth=1.2)
        ax.text(*p_j, '$P_j$', fontsize=30)
        self.__BasicAnnotation(ax,
                               p=p_j_m,
                               p_n='D_j^-',
                               offsets=(-0.1, -0.2),
                               shrink=0.05)

    def GI(self, save_name: str):
        fig, ax = plt.subplots(1, 1, figsize=self.__dims)

        self.__GIPlot(ax)
        fig.tight_layout()
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()

    def __AddArrow(self, ax, p_0: tuple, p_1: tuple):
        common_opts = dict(arrowstyle=u'-|>', lw=1.2)
        arrow = FancyArrowPatch(p_0,
                                p_1,
                                color='black',
                                mutation_scale=20,
                                **common_opts)
        ax.add_patch(arrow)
        return

    def __SIPlot(self, ax: any) -> None:
        self.__ScatterGen(ax)
        self.__LIAnnotation(ax)
        a_ac, a_dc = self.__OffAxisAngle(ax)
        p_sta = np.array([ax.get_xlim(), ax.get_ylim()]).T[0]
        p_end = np.array([ax.get_xlim(), ax.get_ylim()]).T[1]
        theta_li_0 = p_end * 0.1 + p_sta
        theta_li_1 = (p_end[0] * 0.15 + p_sta[0], 0)
        l_theta_li = np.array([theta_li_0, theta_li_1]).T
        theta_li = ((theta_li_0[0] - p_sta[0]) * 1.1 + p_sta[1],
                    (theta_li_0[1] - p_sta[1]) * 0.6 + p_sta[1])
        ax.text(*theta_li, '$\\theta_{LI}$', fontsize=20)
        ax.plot(*l_theta_li, 'k-', lw=0.8)

        # point: P_i(RR_i < RR_{i+1})
        p_i_i = list(a_dc.keys())[1]
        p_i = (self.__arr_0[p_i_i], self.__arr_1[p_i_i])
        ax.text(*p_i, '$P_i$', fontsize=30)
        theta_i_0 = (np.array(p_i) - p_sta) * 0.15 + p_sta
        theta_i_1 = (p_i[0] * 0.2 + p_sta[0], 0)
        l_theta_i = np.array([theta_i_0, theta_i_1]).T
        theta_i = ((theta_i_0[0] - p_sta[0]) * 1.3 + p_sta[1],
                   (theta_i_0[1] - p_sta[1]) * 0.9 + p_sta[1])
        ax.text(*theta_i, '$\\theta_i$', fontsize=20)
        ax.plot(*l_theta_i, 'k-', lw=0.8)
        self.__AddArrow(ax, p_sta, p_i)

        # point: P_j(RR_i > RR_{i+1})
        p_j_i = list(a_ac.keys())[0]
        p_j = (self.__arr_0[p_j_i], self.__arr_1[p_j_i])
        ax.text(*p_j, '$P_j$', fontsize=30)
        theta_j_0 = (np.array(p_j) - p_sta) * 0.55 + p_sta
        theta_j_1 = (p_j[0] * 0.5 + p_sta[0], 0)
        l_theta_j = np.array([theta_j_0, theta_j_1]).T
        theta_j = ((theta_j_0[0] - p_sta[0]) * 1.3 + p_sta[1],
                   (theta_j_0[1] - p_sta[1]) * 0.5 + p_sta[1])
        ax.text(*theta_j, '$\\theta_j$', fontsize=20)
        ax.plot(*l_theta_j, 'k-', lw=0.8)
        self.__AddArrow(ax, p_sta, p_j)

    def SI(self, save_name: str):
        fig, ax = plt.subplots(1, 1, figsize=self.__dims)
        self.__SIPlot(ax)
        fig.tight_layout()
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()

    def CombinePlot(self, save_name: str):
        dim_plus = tuple(i * 2 + 1 for i in self.__dims)
        sup_title_st = dict(family='Arial', style='normal', size=30)
        fig, ((sd, pi), (gi, si)) = plt.subplots(2,
                                                 2,
                                                 figsize=dim_plus,
                                                 constrained_layout=True)
        self.__SDPlot(sd)
        sd.set_title('$(a) SD1, SD2$', y=-0.16, fontdict=sup_title_st)
        self.__PIPlot(pi)
        pi.set_title('$(b) PI$', y=-0.16, fontdict=sup_title_st)
        self.__GIPlot(gi)
        gi.set_title('$(c) GI$', y=-0.16, fontdict=sup_title_st)
        self.__SIPlot(si)
        si.set_title('$(d) SI$', y=-0.16, fontdict=sup_title_st)
        # fig.tight_layout(w_pad=0.5, h_pad=1.2)
        fig.savefig(s_f_fold / (save_name + '.png'), dpi=300)
        plt.close()


def GenLatexPdf(fig_pathes: list):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('booktabs'))

    chart_description = r'''Four methods of variability analysis of the Pongarets. 
    (a) is general Poncare analysis, where SD1 is the degree of variation in the 
    short axis and SD2 is the degree of variation in the long axis. (b) is the asymmetry 
    analysis, which calculates the proportion of points above and below the LI to the 
    total(PI). (c) is the asymmetry analysis, and the sum of the distances from 
    the points above LI to LI is calculated as the proportion of the total(GI). 
    (d) is an asymmetry analysis, which calculates the ratio of the angle between 
    the point above LI and LI to the total(SI).'''

    with doc.create(Figure(position='h!')) as imagesRow1:
        doc.append(Command('centering'))
        with doc.create(
                SubFigure(position='c',
                          width=NoEscape(r'0.5\linewidth'))) as left_row:
            left_row.add_image(str(fig_pathes[0]),
                               width=NoEscape(r'0.95\linewidth'))

        with doc.create(
                SubFigure(position='c',
                          width=NoEscape(r'0.5\linewidth'))) as right_row:
            right_row.add_image(str(fig_pathes[1]),
                                width=NoEscape(r'0.95\linewidth'))

    with doc.create(Figure(position='h!')) as imagesRow2:
        doc.append(Command('centering'))
        with doc.create(
                SubFigure(position='c',
                          width=NoEscape(r'0.5\linewidth'))) as left_row:
            left_row.add_image(str(fig_pathes[2]),
                               width=NoEscape(r'0.95\linewidth'))

        with doc.create(
                SubFigure(position='c',
                          width=NoEscape(r'0.5\linewidth'))) as right_row:
            right_row.add_image(str(fig_pathes[3]),
                                width=NoEscape(r'0.95\linewidth'))

    with doc.create(Paragraph('')) as tail:
        tail.append(bold('Fig.1 '))
        tail.append(NoEscape(chart_description))

    return doc


def annot_value(p: tuple, text, ax=None):

    arrowprops = dict(arrowstyle="->",
                      color='red',
                      connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',
              textcoords='axes fraction',
              arrowprops=arrowprops,
              fontsize=13,
              ha='right',
              va='top')
    ax.annotate(text, xy=(p[0], p[1]), xytext=(p[0], p[1]), **kw)


def points_sd(width: float, height: float, p: tuple, l_axis: np.ndarray,
              s_axis: np.ndarray):
    l_axis = np.array(l_axis).T
    s_axis = np.array(s_axis).T
    l_axis_len = np.sqrt(
        np.square(l_axis[-1][0] - p[0]) + np.square(l_axis[-1][1] - p[1]))
    s_axis_len = np.sqrt(
        np.square(s_axis[-1][0] - p[0]) + np.square(s_axis[-1][1] - p[1]))
    l_shrink = width / l_axis_len
    s_shrink = height / s_axis_len
    sd_1_0 = p[0] + s_shrink * (s_axis[-1][0] - p[0])
    sd_1_1 = p[1] + s_shrink * (s_axis[-1][1] - p[1])
    sd_2_0 = p[0] + l_shrink * (l_axis[-1][0] - p[0])
    sd_2_1 = p[1] + l_shrink * (l_axis[-1][1] - p[1])
    sd_1 = np.array([sd_1_0, sd_1_1])
    sd_2 = np.array([sd_2_0, sd_2_1])
    return sd_1, sd_2


def centroid_connection(a_0: list, a_1: list, multiplier: float = 1.8):
    centroid = np.array([sum(a_0) / len(a_0), sum(a_1) / len(a_1)])
    ex_point = centroid * multiplier
    #   SD1 axis array
    l_axis_a = (np.array([0, centroid[0], ex_point[0]]),
                np.array([0, centroid[1], ex_point[1]]))
    s_axis_p = plumbline_set((0, 0), ex_point, centroid)
    s_axis_p_0_0 = (s_axis_p[0] - centroid[0]) * np.sqrt(0.2) + centroid[0]
    s_axis_p_0_1 = (s_axis_p[1] - centroid[1]) * np.sqrt(0.2) + centroid[1]
    s_axis_p_1_0 = centroid[0] - (s_axis_p[0] - centroid[0]) * np.sqrt(0.2)
    s_axis_p_1_1 = centroid[1] - (s_axis_p[1] - centroid[1]) * np.sqrt(0.2)
    #   SD2 axis array
    s_axis_a = (np.array([s_axis_p_1_0, s_axis_p_0_0]),
                np.array([s_axis_p_1_1, s_axis_p_0_1]))
    return centroid, l_axis_a, s_axis_a


def euclidean_distance(k, h, pointIndex):
    '''
    calculate euclidean distance
    :param k: line Slope (float)
    :param h: line intercept (float)
    :param pointIndex: point location (tuple)
    :return: euclidean distance (float)
    '''
    x = pointIndex[0]
    y = pointIndex[1]
    theDistance = math.fabs(h + k * (x - 0) - y) / (math.sqrt(k * k + 1))
    return theDistance


def plumbline_set(p_1: tuple, p_2: tuple, p_3: tuple):
    '''
    p_1: point 1 in line
    p_2: point 2 in line
    p_3: point in the vertical line of the line
    '''
    if p_2[0] == p_1[0]:
        x = p_1[0]
        y = p_3[0]
    else:
        k, b = np.linalg.solve([[p_1[0], 1], [p_2[0], 1]], [p_1[1], p_2[1]])
        if k * p_3[0] + b == p_3[1]:
            k = -1 / k
            b = p_3[1] - k * p_3[0]
            y = 0
            x = -1 * b / k
        else:
            x = np.divide(((p_2[0] - p_1[0]) * p_3[0] +
                           (p_2[1] - p_1[1]) * p_3[1] - b * (p_2[1] - p_1[1])),
                          (p_2[0] - p_1[0] + k * (p_2[1] - p_1[1])))
            y = k * x + b

    return x, y


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    x = np.array(x)
    y = np.array(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse, ell_radius_x * scale_x, ell_radius_y * scale_y


if __name__ == '__main__':
    main()