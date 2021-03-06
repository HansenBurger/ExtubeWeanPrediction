import sys
from pathlib import Path
from datetime import datetime
from data import StaticData

sys.path.append(str(Path.cwd()))

from Classes.ExtractSplice import ExtractSplice
from Classes.TypesInstant import RecordInfo
from Classes.Func.KitTools import ConfigRead, measure

static = StaticData()
mode_info = static.mode_info
para_t_st = static.verify_st['para_t_st']
para_n_st = static.verify_st['para_n_st']

# G-5 ResampleRate: 256 -- 6776243


@measure
def main(mode_: str) -> None:

    data_path = Path(ConfigRead('WaveData', mode_))
    query_list = RecQuery(mode_)

    def RidInit(que_o: any) -> any:
        '''
        '''
        main_p = RecordInfo(que_o.rid, que_o.e_t)
        main_p.ParametersInit(data_path, que_o.opt)
        reco_p = ExtractSplice(main_p.rec)
        reco_p.RecBatchesExtract([que_o.zdt], [que_o.rec_t])
        reco_p.ParaSelecting(para_t_st, para_n_st)

        return main_p.rec

    def InfoCollect(rid_o: any) -> dict:
        '''
        '''
        machine_name = rid_o.vm_n

        wave = rid_o.waves[0] if rid_o.waves[0] else None
        para = rid_o.paras[0] if rid_o.paras[0] else None

        if not wave:
            resp_val_t = None
        else:
            resp_val_l = [i.wid if i.val else 0 for i in wave.resps]
            resp_val_t = round(sum(resp_val_l))

        if not para:
            vent_mode_d = None
            peep_ps_sum_d = None
        else:
            vent_mode_d = para.st_mode
            peep_ps_sum_d = {}
            for st_t in para_t_st:
                st_t_ps = para.st_ps[st_t]
                st_t_peep = para.st_peep[st_t]

                if st_t_ps == None or st_t_peep == None:
                    peep_ps_sum_d[st_t] = None
                else:
                    peep_ps_sum_d[st_t] = st_t_ps + st_t_peep

        return {
            'v_t': resp_val_t,
            'mch': machine_name,
            'vmd': vent_mode_d,
            'spd': peep_ps_sum_d
        }

    for que_o in query_list:
        t_s = datetime.now()
        if not que_o.zdt or not que_o.zpx:
            continue
        else:
            rid_o_contain = RidInit(que_o)
            rec_i_collect = InfoCollect(rid_o_contain)
            for col_n in rec_i_collect.keys():
                setattr(que_o, col_n, rec_i_collect[col_n])
            que_o.save()
        t_e = datetime.now()
        print('Process row: {0}, Consume: {1}'.format(que_o.index, t_e - t_s))


def RecQuery(q_mode: str, index_range: range = None):
    dst_class = mode_info[q_mode]['class']
    if not index_range:
        query_list = dst_class.select()
    else:
        cond_0 = dst_class.index >= index_range[0]
        cond_1 = dst_class.index <= index_range[-1]
        query_list = dst_class.select().where(cond_0 & cond_1)

    return query_list


if __name__ == '__main__':
    main(sys.argv[1])
