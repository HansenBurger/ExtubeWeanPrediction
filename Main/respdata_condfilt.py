#TODO following
'''
Three things
1. save basic filting result(which include):
    BY MODE = [Extube, Wean]
    --- respdata query ----
    1. Total p, Exist p (Succ | Fail)
    2. valid p (Succ | Fail)
    3. invalid p (Succ | Fail)
        1. p cant find rid (no rid in op-day | rid no route)
        2. p cant find any rec in rid (zdt or zpx not exist)
    ---- respdata verify ----
    1. valid p (Succ | Fail)
    2. invalid p (Succ | Fail)
        1. wave info not exist (may not use)
        2. machine name is '840-4' or ''840-22'
        3. vent time less than 600s (may not use)

2. query data
    1. generate pid gp
    2. extube/wean time re-ensure
        1. seperate dataframe by op_end time
        2. check if psv mode in df_down(change time set)
        3.  
'''
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from functools import reduce
from Classes.Func.KitTools import measure, ConfigRead
from Classes.ORM.basic import OutcomeExWean, ExtubePrep, WeanPrep
from Classes.ORM.cate import ExtubePSV, ExtubeSumP12

main_mode = 'Wean'
mode_info = {
    'Extube': {
        'class': ExtubePrep,
        'd_e_s': OutcomeExWean.we_s,
        'd_e_t': OutcomeExWean.we_t
    },
    'Wean': {
        'class': WeanPrep,
        'd_e_s': OutcomeExWean.ex_s,
        'd_e_t': OutcomeExWean.ex_t
    }
}


@measure
def main():
    p = FiltInfoCollect(main_mode)
    print(p.InitialDst())
    pass


def RecQuery(q_mode: str, save_path: Path):
    pass

    # TODO extube in src_1, total


class FiltInfoCollect():
    def __init__(self, mode_: str):
        self.__mode = mode_
        self.__src_0 = OutcomeExWean
        self.__src_1 = mode_info[mode_]['class']

    def __PosNegCond(self, col: any):
        cond_0 = col.contains('成功')
        cond_1 = col.contains('失败')

        return cond_0, cond_1

    def InitialDst(self):

        end_i = mode_info[self.__mode]['d_e_s']
        cond = end_i != None
        cond_0, cond_1 = self.__PosNegCond(end_i)

        que_ = self.__src_0.select()

        cond_l = [cond, cond_0]

        a = reduce(lambda x, y: x & y, cond_l)

        tot_size = len(que_.where(cond))
        succ_size = len(que_.where(a))
        fail_size = len(que_.where(cond_1 & cond))

        info = 'Total: {0}, succ: {1} | fail: {2}'.format(
            tot_size, succ_size, fail_size)

        print(type(end_i))

        return info

    def __PartialDst(self):
        src_0, src_1 = self.__src_0, self.__src_1
        end_i = self.__dict_[self.__mode]

        join_d = {
            'dest': src_0,
            'on': (src_0.pid == src_1.pid),
            'attr': 'binfo'
        }
        gp_col = [src_1.pid]
        cond_0, cond_1 = self.__PosNegCond(src_1.e_s)
        cond_rot = ~src_1.rot.is_null()
        cond_rec = ~src_1.zdt.is_null() & ~src_1.zpx.is_null()

        que_ = src_1.select().join(**join_d)

        tot_size = len(src_0.select().where(~end_i.is_null()))
        mat_size = len(que_.group_by(*gp_col))
        val_len0 = len(que_.where(cond_rot).group_by(*gp_col))
        val_len1 = len(que_.where(cond_rot & cond_rec).group_by(*gp_col))


if __name__ == '__main__':
    main()