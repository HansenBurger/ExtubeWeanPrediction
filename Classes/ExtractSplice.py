import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from operator import mul, sub, add
from Classes.TypesInstant import RecordResp, RecordPara
from Classes.Func.KitTools import LocatSimiTerms, GetObjectDict


class basic():
    def __init__(self) -> None:
        pass


class ExtractSplice(basic):
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
        """
        RecBatchesExtract

        Collect the RidRec-class obj's waves and paras data

        id_list: zdt, zpx id in positive order
        t_set: time range set before op-end
        """
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
            id_list.reverse()
            for i in id_list:
                if v_still_t > t_set:
                    break
                w_d = self.__WaveBatchGen(i)
                p_d = self.__ParaBatchGen(i)
                wave_data.append(w_d)
                para_data.append(p_d)
                v_still_t += sum([i.wid for i in w_d.resps if i.val])

            if v_still_t > t_set or v_still_t == t_set:
                pass
            else:
                wave_data = []
                para_data = []

            wave_data.reverse()
            para_data.reverse()

        else:
            print('Wrong type of t_set')
            return

        self.__ridrec.waves = wave_data
        self.__ridrec.paras = para_data

    def ParaSplicing(self, t_set_s):
        """
        ParaSplicing
        
        Main function to get the parainfo by given time points, the
        INPUT are para-class objs sequance which order in positive
        (by record time), the OUTPUT mainly composed by two parts. First,
        the para-class contain para list changing to dict about time-points
        info. Second, return a dict of all types parainfo in reverse order
        (from op-tail to op-head)
        output  
        
        t_set_s: Time points for querying para-info (unit second)
        """
        # Positive order in (head to tail)
        para_dict = {}

        para_data = self.__ridrec.paras
        vm_sr = self.__ridrec.waves[0].sr
        p_select = GetObjectDict(para_data[0])
        del p_select['zpx'], p_select['u_ind']  # del para type not interested

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

        # Reverse order out (tail to head)
        return para_dict

    def RespSplicing(self, t_set, vm_cond):
        """
        RespSplicing

        Main fucn for getting Valid-Resps in specific time range before op-end
        INPUT are para-class objs and wave-class objs in positive order
        OUTPUT are resp-class objs in positive order from t_set range before op-end

        t_set: Time set range(unit second) before op-end
        vm_cond: The vent-mode list full fill require(PSV)
        """
        # Positive order in (head to tail)
        resp_select = []

        para_data = self.__ridrec.paras.copy()
        wave_data = self.__ridrec.waves.copy()

        if not wave_data or not para_data:
            return None

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
            ut_s.extend(ut_l)

        ut_s = list(map(mul, ut_s, [-1] * len(ut_s)))
        vm_l = vm_l[0:LocatSimiTerms(ut_s, [t_set])[t_set]]
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

        # Positive order (range head to op-tail)
        return resp_select