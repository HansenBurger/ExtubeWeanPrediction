import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path.cwd()))

from Classes.ExtractSplice import ExtractSplice
from Classes.TypesInstant import RecordInfo
from Classes.ORM.main import ExtubePrep, WeanPrep
from Classes.Func.KitTools import ConfigRead, measure

main_mode = 'Extube'
mode_info = {'Extube': {'class': ExtubePrep}, 'Wean': {'class': WeanPrep}}


@measure
def main() -> None:

    st_t_ran = [10, 900, 1800, 2700]
    st_paras = ['st_mode', 'st_peep', 'st_ps']

    data_path = Path(ConfigRead('WaveData', main_mode))
    query_list = RecQuery(main_mode)

    def RidInit(que_o: any) -> any:
        '''
        '''
        main_p = RecordInfo(data_path, que_o.e_t, que_o.rid)
        main_p.ParametersInit()
        rid_obj = main_p.rec
        reco_p = ExtractSplice(rid_obj)
        reco_p.RecBatchesExtract([que_o.zdt])
        reco_p.ParaSelecting(st_t_ran, st_paras)

        return rid_obj

    def InfoCollect(rid_o: any) -> dict:
        '''
        '''
        machine_name = rid_o.vm_n

        wave = rid_o.waves[0] if rid_o.waves[0] else None
        para = rid_o.paras[0] if rid_o.paras[0] else None

        if not wave:
            resp_val_t = None
        else:
            resp_val_l = [i.wid if i.wid else 0 for i in wave.resps]
            resp_val_t = round(sum(resp_val_l))

        if not para:
            vent_mode_d = None
            peep_ps_sum_d = None
        else:
            vent_mode_d = para.st_mode
            peep_ps_sum_d = {}
            for st_t in st_t_ran:
                st_t_ps = para.st_ps[st_t]
                st_t_peep = para.st_peep[st_t]

                if not st_t_ps or not st_t_peep:
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


def RecQuery(q_mode: str, index_slice: slice = None):
    dst_class = mode_info[q_mode]['class']
    if not index_slice:
        query_list = dst_class.select()
    else:
        query_list = dst_class.select()[index_slice]

    return query_list


if __name__ == '__main__':
    main()