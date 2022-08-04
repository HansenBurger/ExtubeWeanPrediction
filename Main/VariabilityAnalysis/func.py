import sys
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from datetime import datetime
from data import StaticData, DynamicData

sys.path.append(str(Path.cwd()))

from Classes.Domain import layer_p
from Classes.TypesInstant import RecordInfo
from Classes.ExtractSplice import ExtractSplice
from Classes.VarResultsGen import VarResultsGen
from Classes.Func.KitTools import ConfigRead, FromkeysReid
from Classes.Func.DiagramsGen import PlotMain
from Classes.Func.CalculatePart import SenSpecCounter, PerfomAssess
from FlowCalibration.incators_calculate import RespValStatic, PoincareScatter

static = StaticData()
dynamic = DynamicData()


def LocInit(s_f: Path, s_g: Path, mode_name: str):
    '''
    '''
    mode_i = mode_name.split('_')
    dynamic.data_loc = Path(ConfigRead('WaveData', mode_i[0]))
    dynamic.s_f_fold = s_f / mode_name
    dynamic.s_f_fold.mkdir(parents=True, exist_ok=True)
    dynamic.s_g_fold = s_g / mode_name
    dynamic.s_g_fold.mkdir(parents=True, exist_ok=True)
    dynamic.pid_dr_s = {}


def TableQuery(mode_name: str, mv_t_ran: tuple = (48, 2160)):
    '''
    '''
    mode_i = mode_name.split('_')
    src_0 = static.cate_info[mode_i[0]][mode_i[1]]

    # src_2 = static.op_basic_i

    def NonAgedIll(src_1: any) -> list:
        '''
        '''
        join_info = {
            'dest': src_1,
            'on': src_1.pid == src_0.pid,
            'attr': 'pinfo'
        }
        col_rmk = [
            src_1.rmk, src_1.rmk_i, src_1.rmk_i_, src_1.rmk_o, src_1.rmk_o_
        ]
        c_age = src_1.age <= 75
        c_d = lambda x: (x.is_null()) | (~x.contains('脑') & ~x.contains('神经'))
        c_d_rmk = reduce(lambda x, y: x & y, [c_d(col) for col in col_rmk])
        que_l = src_0.select(src_0.pid).join(**join_info).where(c_age
                                                                & c_d_rmk)
        pid_l = [que.pid for que in que_l.group_by(src_0.pid)]
        return pid_l

    def MVRange(src_1: any) -> list:
        '''
        '''
        join_info = {
            'dest': src_1,
            'on': src_1.pid == src_0.pid,
            'attr': 'pinfo'
        }
        c_mv = (src_1.mv_t >= mv_t_ran[0]) & (src_1.mv_t <= mv_t_ran[1])
        que_l = src_0.select(src_0.pid).join(**join_info).where(c_mv)
        pid_l = [que.pid for que in que_l.group_by(src_0.pid)]
        return pid_l

    c_pid_test = src_0.pid > 0
    c_Nad = src_0.pid.in_(NonAgedIll(
        static.p_basic_i)) if len(mode_i) > 2 else src_0.pid > 0
    c_Nrid = ~src_0.pid.in_(static.cate_info[mode_i[0]]['multirid'])
    c_mv_t = src_0.pid.in_(MVRange(static.op_basic_i))

    que = src_0.select().where(c_Nad & c_Nrid & c_mv_t & c_pid_test)

    df = pd.DataFrame(list(que.dicts()))
    df = df.drop('index', axis=1)
    df.e_s = np.where(df.e_s.str.contains('成功'), 0, 1)
    gp = df.groupby('pid')

    for pid in df.pid.unique():
        df_tmp = gp.get_group(pid)
        df_tmp = df_tmp.reset_index(drop=True)

        pid_dr = {}
        pid_dr['df_in'] = df_tmp
        dynamic.pid_dr_s[pid] = pid_dr


def PidVarCount(t_set: int, pid_s: list = []):
    '''
    '''
    def DataGen(df: pd.DataFrame) -> layer_p.Patient:
        '''
        '''
        rid = df.rid.unique()[-1]

        pid_o = layer_p.Patient()
        pid_o.pid = df.pid[0]
        pid_o.icu = df.icu[0]
        pid_o.end_t = df.e_t[0]
        pid_o.end_i = df.e_s[0]

        rid_p = RecordInfo(rid, pid_o.end_t)
        rid_p.ParametersInit(dynamic.data_loc, df.opt[0])
        pid_o.rid_s = rid_p.rec

        rec_id_s = df.zdt.tolist()
        rec_t_s = df.rec_t.tolist()

        splice_p = ExtractSplice(pid_o.rid_s)
        splice_p.RecBatchesExtract(rec_id_s, rec_t_s, t_set)
        pid_o.resp_l, pid_o.validy = splice_p.RespSplicing(
            static.psv_vms, t_set)

        return pid_o

    pid_dr_s = {k: v
                for k, v in dynamic.pid_dr_s.items()
                if v in pid_s} if pid_s else dynamic.pid_dr_s

    for pid, dr in pid_dr_s.items():

        t_s = datetime.now()
        pid_obj = DataGen(dr['df_in'])
        dynamic.pid_dr_s[pid]['end'] = pid_obj.end_i
        dynamic.pid_dr_s[pid]['validy'] = pid_obj.validy

        if not pid_obj.resp_l:
            print('{0}\' has no valid data'.format(pid))
            continue
        else:
            resp_ind_save = dynamic.s_g_fold / 'IndInfo' / str(pid_obj.pid)
            resp_ind_save.mkdir(parents=True, exist_ok=True)
            # pid_obj.resp_l = RespValStatic(pid_obj.resp_l, resp_ind_save)
            process_1 = VarResultsGen(pid_obj)
            process_1.VarRsGen(static.methods)
            process_1.TensorStorage(dynamic.s_f_fold)
            t_e = datetime.now()
            print('{0}\'s data consume {1}'.format(pid, (t_e - t_s)))


def VarInfoCollect():
    save_path = dynamic.s_f_fold / 'process_info.txt'
    save_param = {
        'val_succ_n': 0,
        'val_fail_n': 0,
        'data_inval': 0,
        'mode_inval': 0,
        'mode_inval_succ': 0,
        'mode_inval_fail': 0
    }
    repr_gen = lambda dict_: ('\n').join(k + ':\t' + str(v)
                                         for k, v in dict_.items())
    for p_dr in dynamic.pid_dr_s.values():
        p_dr_val = reduce(lambda x, y: x & y, list(p_dr['validy'].values()))

        end_con_0 = not p_dr['end']
        end_con_1 = p_dr['end']
        data_con = sum(p_dr['validy'].values()) == 2
        mode_con = sum(p_dr['validy'].values()) == 1

        if p_dr_val:
            save_param['val_succ_n'] += 1 if end_con_0 else 0
            save_param['val_fail_n'] += 1 if end_con_1 else 0
            save_param['val_ratio'] = 0
        else:
            save_param['data_inval'] += 1 if data_con else 0
            save_param['mode_inval'] += 1 if mode_con else 0
            save_param['mode_inval_succ'] += 1 if mode_con and end_con_0 else 0
            save_param['mode_inval_fail'] += 1 if mode_con and end_con_1 else 0

    save_param['val_ratio'] = save_param['val_fail_n'] / (
        save_param['val_succ_n'] + save_param['val_fail_n'])
    param_repr = repr_gen(save_param)

    with open(save_path, 'w') as f:
        f.write('ProcessInfo:\n')
        f.write(param_repr)


def VarStatistics() -> list:
    '''
    '''
    p_r_l, p_i_l = [], []
    file_loc = dynamic.s_f_fold
    for path in Path(file_loc).iterdir():
        if not path.is_file() or path.suffix != '.csv':
            pass
        else:
            p_r_l.append(pd.read_csv(path, index_col='method'))
            p_info = path.name.split('_')
            p_i_d = {'pid': p_info[0], 'end': int(p_info[1]), 'rid': p_info[2]}
            p_i_l.append(p_i_d)

    p_i_df = pd.DataFrame(p_i_l)

    methods = p_r_l[0].index
    indicat = p_r_l[0].columns
    p_r_d = FromkeysReid(methods, {})
    roc_df = p_r_l[0].copy()
    p_v_df = p_r_l[0].copy()
    tot_df = pd.DataFrame()

    for i in methods:
        for j in indicat:
            array_ = np.array([x.loc[i, j] for x in p_r_l])
            process_ = PerfomAssess(p_i_df.end, array_)
            auc, _, _ = process_.AucAssess()
            p, _, _ = process_.PValueAssess()
            roc_df.loc[i, j] = auc
            p_v_df.loc[i, j] = p
            tot_df['-'.join([i, j])] = array_

            p_r_d[i][j] = array_

    df_pos, df_neg = NegPosGet(p_i_df, p_r_d)
    df_fp, df_fn, _, _ = FalseBuild(p_i_df, p_r_d)
    corr_p = tot_df.corr(method='pearson')
    corr_s = tot_df.corr(method='spearman')
    corr_k = tot_df.corr(method='kendall')

    s_g_fold = dynamic.s_g_fold
    plot_p = PlotMain(s_g_fold)

    def ChartsSave(df: pd.DataFrame,
                   save_n: str,
                   chart_type: any,
                   chart_kwags: dict = {}) -> None:
        df.to_csv(s_g_fold / (save_n + '.csv'), index=False)
        chart_type(df, save_n, **chart_kwags)

    ChartsSave(roc_df, 'AUC_HeatMap', plot_p.HeatMapPlot, {'cmap': 'coolwarm'})
    ChartsSave(p_v_df, 'P_HeatMap', plot_p.HeatMapPlot, {'cmap': 'YlGnBu'})
    ChartsSave(corr_p, 'Corr_Pearson', plot_p.HeatMapLarge,
               {'cmap': 'coolwarm'})
    ChartsSave(corr_k, 'Corr_Kendall', plot_p.HeatMapLarge,
               {'cmap': 'coolwarm'})
    ChartsSave(corr_s, 'CorrSpearman', plot_p.HeatMapLarge,
               {'cmap': 'coolwarm'})
    ChartsSave(df_pos, 'RSBI_fail_med', plot_p.SensSpecPlot)
    ChartsSave(df_neg, 'RSBI_succ_med', plot_p.SensSpecPlot)

    df_fp.to_csv(s_g_fold / 'RSBI_105_fp.csv', index=False)
    df_fn.to_csv(s_g_fold / 'RSBI_105_fn.csv', index=False)


def NegPosGet(df, dict_):
    df_ = pd.DataFrame()
    df_['end_0'] = df['end']
    df_['end_1'] = ~np.array(df['end'].tolist()) + 2
    df_['rsbi'] = dict_['med']['rsbi']
    cut_arr = np.linspace(0, 199, 399)

    l_d_pos, l_d_neg = [], []

    for i in cut_arr:
        process_ = SenSpecCounter(i, df_.rsbi)
        dict_0 = process_.CutEndPos(df_.end_0, np.int8)
        dict_1 = process_.CutEndNeg(df_.end_1, np.int8)
        l_d_pos.append(dict_0)
        l_d_neg.append(dict_1)

    df_pos = pd.DataFrame(l_d_pos)
    df_neg = pd.DataFrame(l_d_neg)
    return df_pos, df_neg


def FalseBuild(df: pd.DataFrame, dict_: dict):
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