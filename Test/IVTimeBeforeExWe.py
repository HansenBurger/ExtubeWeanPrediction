import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

fold = Path(r'C:\Main\Data\_\Form\PatientData\20220613')
file_ = [
    'Ex_Wean_Outcome.csv', 'PatientInfo.csv', 'Extube_Patient_Info.csv',
    'Wean_Patient_Info.csv'
]

df = pd.read_csv(fold / file_[0])
mode_col = ['ExtubeStatus_psv', 'WeanStatus_psv']


def main():
    cate_set = {
        'Hour': {
            'size': (20, 10),
            'name': 'HourDist'
        },
        'Day': {
            'size': (18, 10),
            'name': 'DayDist'
        },
        'Month': {
            'size': (10, 5),
            'name': 'MonthDist'
        },
    }
    for mode in mode_col:
        save_p = fold / mode
        save_p.mkdir(parents=True, exist_ok=True)
        cate_dfs = GetCateByHours(df, mode)

        for cate in cate_set.keys():
            cate_i = cate_set[cate]
            cate_i.update(cate_dfs[cate])
            DistResultGen(cate_i['data'], cate_i['size'], save_p,
                          cate_i['name'])


def GetCateByHours(df: pd.DataFrame, type_col: str) -> str:
    type_col = type_col.split('_')[0]
    mv_st = df[~df[type_col].isnull()]
    mv_st = mv_st[['PID', 'MVlastTime', type_col]]
    mv_st = mv_st.rename({
        'PID': 'pid',
        'MVlastTime': 'mv_t',
        type_col: 'end'
    },
                         axis=1)
    mv_st = mv_st[~mv_st.end.str.contains('其他模式')]
    mv_st['end'] = [0 if '成功' in i else 1 for i in mv_st.end]
    mv_st['cate'] = pd.Series([], dtype='str')

    cate_info = {
        'Hour': {
            'head': 'H_',
            'cond': mv_st.mv_t < 24,
            'divisor': 1,
            'data': pd.DataFrame()
        },
        'Day': {
            'head': 'D_',
            'cond': (mv_st.mv_t >= 24) & (mv_st.mv_t < 744),
            'divisor': 24,
            'data': pd.DataFrame()
        },
        'Month': {
            'head': 'M_',
            'cond': mv_st.mv_t >= 744,
            'divisor': 744,
            'data': pd.DataFrame()
        }
    }

    for cate_ in cate_info.keys():
        cate_i = cate_info[cate_]
        mv_st_c = mv_st.loc[cate_i['cond']]
        series = (mv_st.mv_t / cate_i['divisor']).round().astype('int')
        series = cate_i['head'] + series.astype('str').str.rjust(2, '0')
        mv_st_c.cate = series
        mv_st_c.sort_values(by=['cate', 'pid'], inplace=True)
        mv_st_c.reset_index(drop=True, inplace=True)
        cate_i['data'] = mv_st_c

    return {k: {'data': v['data']} for k, v in cate_info.items()}


def DistResultGen(df: pd.DataFrame, fig_size: tuple, save_path: Path,
                  save_name: str):
    sns.reset_orig()
    sns.set_style(style='whitegrid')
    plt.subplots(figsize=fig_size)
    sns.histplot(df, x='cate', hue='end', multiple='stack', shrink=.9)
    plt.xlabel('MV Time (H/D/M)', fontdict={'fontsize': 15})
    plt.ylabel('Count (n)', fontdict={'fontsize': 15})
    plt.title('Distribution of ventilation time before Wean/Extubate',
              fontdict={'fontsize': 20})
    plt.tight_layout()
    plt.savefig(str(save_path / (save_name + '.png')))
    plt.close()

    with open(save_path / (save_name + '.txt'), 'w') as f:
        f.write(save_name + ': \n')
        for cate in df.cate.unique().tolist():
            cond_c = df.cate == cate
            cond_0, cond_1 = df.end == 0, df.end == 1
            succ_ = df[cond_c & cond_0].shape[0]
            fail_ = df[cond_c & cond_1].shape[0]
            f.write('{0}: succ | fail = {1} | {2}'.format(cate, succ_, fail_))
            f.write('\n')


if __name__ == '__main__':
    main()
