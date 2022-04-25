import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path.cwd()))
from Classes.Func import CalculatePart, DiagramsGen, KitTools

mode_name = 'wean_psv'


def main():
    var_rs_f = r'C:\Main\Data\_\Result\Form\20220423_20_wean_psv'
    var_rs_p = Path(KitTools.ConfigRead('ResultSave', 'Form')) / var_rs_f
    s_g_fold = KitTools.SaveGen(KitTools.ConfigRead('ResultSave', 'Graph'),
                                mode_name)
    p_i_df, p_r_d, roc_df = DataCombine(var_rs_p)
    df_pos, df_neg = NegPosGet(p_i_df, p_r_d)
    df_fp, df_fn, _, _ = FalseBuild(p_i_df, p_r_d)

    plot_p = DiagramsGen.PlotMain(s_g_fold)

    plot_p.HeatMapPlot(roc_df, 'AUC HeatMap')
    plot_p.SensSpecPlot(df_pos, 'RSBI_fail_med')
    plot_p.SensSpecPlot(df_neg, 'RSBI_succ_med')
    pd.DataFrame.to_csv(df_pos, s_g_fold / 'RSBI_fail_med.csv', index=False)
    pd.DataFrame.to_csv(df_neg, s_g_fold / 'RSBI_succ_med.csv', index=False)
    pd.DataFrame.to_csv(df_fp, s_g_fold / 'RSBI_105_fp.csv', index=False)
    pd.DataFrame.to_csv(df_fn, s_g_fold / 'RSBI_105_fn.csv', index=False)


def DataCombine(file_loc):
    p_r_l = []
    p_i_l = []

    for path in Path(file_loc).iterdir():
        if not path.is_file():
            pass
        else:
            p_r_l.append(pd.read_csv(path, index_col='method'))
            p_info = path.name.split('_')
            p_i_d = {'pid': p_info[0], 'end': int(p_info[1]), 'rid': p_info[2]}
            p_i_l.append(p_i_d)

    p_i_df = pd.DataFrame(p_i_l)

    methods = p_r_l[0].index
    indicat = p_r_l[0].columns
    p_r_d = KitTools.FromkeysReid(methods, {})
    roc_df = p_r_l[0].copy()

    for i in methods:
        for j in indicat:
            array_ = np.array([x.loc[i, j] for x in p_r_l])
            roc = roc_auc_score(p_i_df.end, array_)
            roc_df.loc[i, j] = roc
            p_r_d[i][j] = array_

    return p_i_df, p_r_d, roc_df


def NegPosGet(df, dict_):
    df_ = pd.DataFrame()
    df_['end_0'] = df['end']
    df_['end_1'] = ~np.array(df['end'].tolist()) + 2
    df_['rsbi'] = dict_['med']['rsbi']
    cut_arr = np.linspace(0, 199, 399)

    l_d_pos, l_d_neg = [], []

    for i in cut_arr:
        process_ = CalculatePart.SenSpecCounter(i, df_.rsbi)
        dict_0 = process_.CutEndPos(df_.end_0, np.int8)
        dict_1 = process_.CutEndNeg(df_.end_1, np.int8)
        l_d_pos.append(dict_0)
        l_d_neg.append(dict_1)

    df_pos = pd.DataFrame(l_d_pos)
    df_neg = pd.DataFrame(l_d_neg)
    return df_pos, df_neg


def FalseBuild(df, dict_):
    df['rsbi'] = dict_['med']['rsbi']
    rsbi_pos = df.rsbi > 105
    rsbi_neg = df.rsbi < 105
    end_succ = df.end == 0
    end_fail = df.end == 1
    df_fp = df.loc[rsbi_pos & end_succ]
    df_fn = df.loc[rsbi_neg & end_fail]
    df_tp = df.loc[rsbi_pos & end_fail]
    df_tn = df.loc[rsbi_neg & end_succ]
    print('TP: {0}, FP: {1}, TN: {2}, FN: {3}'.format(df_tp.shape[0],
                                                      df_fp.shape[0],
                                                      df_tn.shape[0],
                                                      df_fn.shape[0]))
    return df_fp, df_fn, df_tp, df_tn


if __name__ == '__main__':
    main()