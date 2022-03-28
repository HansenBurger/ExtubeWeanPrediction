from operator import mul, sub, add
from TypesInstant import RecordResp, RecordPara
from Classes.Func.KitTools import LocatSimiTerms, GetObjectDict


class basic():
    def __init__(self) -> None:
        pass


class Main(basic):
    def __init__(self, ridrec_o):
        super().__init__()
        self.__ridrec = ridrec_o

    def __WaveBatchGen(self, id_):
        ridrec = self.__ridrec
        wav_p = RecordResp(ridrec.zif.parent, id_)
        wav_p.WaveformInit()
        wav_p.IndicatorCalculate()
        return wav_p.rec

    def __ParaBatchGen(self, id_):
        ridrec = self.__ridrec
        vm = ridrec.vm_n
        par_p = RecordPara(ridrec.zif.parent, id_)
        par_p.ParametersInit(vm)
        return par_p.rec

    def RecBatchesExtract(self, id_list, t_set=None):
        # t_set's unit in seconds
        ridrec = self.__ridrec
        wave_data = []
        para_data = []
        if not t_set:
            for i in id_list:
                w_d = self.__WaveBatchGen(i)
                p_d = self.__ParaBatchGen(i)
                wave_data.append(w_d)
                para_data.append(p_d)
        elif type(t_set) == int:
            v_still_t = 0
            for i in id_list:
                if v_still_t > t_set:
                    break
                w_d = self.__WaveBatchGen(i)
                p_d = self.__ParaBatchGen(i)
                wave_data.append(w_d)
                para_data.append(p_d)

            if v_still_t > t_set or v_still_t == t_set:
                pass
            else:
                wave_data = []
                para_data = []

        else:
            print('Wrong type of t_set')
            return

        ridrec.waves = wave_data
        ridrec.paras = para_data

    def ParaSplicing(self, t_set_s):
        # Preface order
        para_dict = {}

        para_data = self.__ridrec.paras
        vm_sr = self.__ridrec.waves[0].sr
        p_select = GetObjectDict(para_data[0])
        del p_select['zpx'], p_select['u_ind']

        for p in para_data:
            ut_l = list(map(mul, p.u_ind, [1 / vm_sr] * len(p.u_ind)))
            ut_s_d = LocatSimiTerms(ut_l, t_set_s)
            for p_type in p_select:
                p_t_d = {k: getattr(p, p_type)[v] for k, v in ut_s_d.items()}
                setattr(p, p_type, p_t_d)

        for p_type in p_select:
            p_t_l = [getattr(p, p_type).values() for p in para_data]
            p_t_l.reverse()
            para_dict[p_type] = p_t_l

        return para_dict

    def RespSplicing(self, t_set, vm_cond):
        # Reverse order
        resp_select = []

        para_data = self.__ridrec.paras
        wave_data = self.__ridrec.waves
        sr = wave_data[0].sr
        para_data.reverse()
        wave_data.reverse()
        ut_s = []  # save uiindexdata of paras
        vm_l = []  # save ventmode of paras
        val_t = lambda x, y: sum([True if i in x else False for i in y])

        for p in para_data:
            p.st_mode.reverse()
            vm_l.extend(p.st_mode)
            p.u_ind.reverse()
            ut_l = list(map(mul, p.u_ind, [1 / sr] * len(p.u_ind)))
            usize = len(ut_l)
            ut_l = list(map(sub, ut_l, [ut_l[0]] * usize))
            ut_l = list(map(add, ut_l, [ut_s[-1]] * usize)) if ut_s else ut_l

        vm_l = vm_l[0:LocatSimiTerms(ut_s, [-t_set])[-t_set]]
        vm_val = [True if val_t(vm, vm_cond) > 0 else False for vm in vm_l]

        if False in vm_val:
            pass
        else:
            v_still_t = 0
            for wave in wave_data:
                wave.resps.reverse()
                for resp in wave.resps:
                    if v_still_t > t_set:
                        break
                    elif not resp.val:
                        continue
                    else:
                        v_still_t += resp.wid
                        resp_select.append(resp)

        return resp_select
