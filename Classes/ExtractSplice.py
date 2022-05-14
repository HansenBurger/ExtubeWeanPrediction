import sys
from pathlib import Path
from functools import reduce

sys.path.append(str(Path.cwd()))

from operator import mul, sub, add
from Classes.TypesInstant import RecordResp, RecordPara
from Classes.Func.KitTools import LocatSimiTerms, FromkeysReid


class basic():
    def __init__(self) -> None:
        pass


class ExtractSplice(basic):
    def __init__(self, ridrec_o):
        super().__init__()
        self.__ridrec = ridrec_o

    def __WaveBatchGen(self, id_, t_):
        ridrec = self.__ridrec
        wav_p = RecordResp(id_, t_)
        wav_p.WaveformInit(ridrec.zif.parent)
        wav_p.IndicatorCalculate()
        return wav_p.rec

    def __ParaBatchGen(self, id_, t_):
        ridrec = self.__ridrec
        vm = ridrec.vm_n
        par_p = RecordPara(id_, t_)
        par_p.ParametersInit(ridrec.zif.parent, vm)
        return par_p.rec

    def RecBatchesExtract(self,
                          id_l: list,
                          t_l: list,
                          t_set: int = -1) -> None:
        """
        **id_l and t_l must be same length**
        Collect the RidRec-class obj's waves and paras data

        id_list: zdt, zpx id in positive order
        t_list: rec time in positive order
        t_set: time range set before op-end
        """
        wave_data = []
        para_data = []
        if type(t_set) == int and t_set < 0:
            for i in range(len(id_l)):
                w_d = self.__WaveBatchGen(id_l[i], t_l[i])
                p_d = self.__ParaBatchGen(id_l[i], t_l[i])
                wave_data.append(w_d)
                para_data.append(p_d)
        elif type(t_set) == int and t_set > 0:
            v_still_t = 0
            id_l.reverse()
            t_l.reverse()
            for i in range(len(id_l)):
                if v_still_t > t_set:
                    break
                w_d = self.__WaveBatchGen(id_l[i], t_l[i])
                p_d = self.__ParaBatchGen(id_l[i], t_l[i])
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

    def __UiIndexSplicing(self) -> list:
        para_data = self.__ridrec.paras.copy()
        wave_data = self.__ridrec.waves.copy()

        sr = wave_data[0].sr
        ut_s = []  # save uiindexdata of paras

        if not sr:
            return ut_s

        for p in reversed(para_data):
            ui_l = list(reversed(p.u_ind))
            ut_l = list(map(mul, ui_l, [1 / sr] * len(ui_l)))
            usize = len(ut_l)
            ut_l = list(map(sub, ut_l, [ut_l[0]] * usize))
            ut_l = list(map(add, ut_l, [ut_s[-1]] * usize)) if ut_s else ut_l
            ut_s.extend(ut_l)

        ut_s = list(map(mul, ut_s, [-1] * len(ut_s)))
        ut_s[0] = 0

        return ut_s

    def ParaSelecting(self, t_set_s, para_attr_l: list):
        """
        ParaSelecting
        
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
        sr = self.__ridrec.waves[0].sr

        for p in para_data:
            ui_l = p.u_ind
            ut_l = list(map(mul, ui_l, [1 / sr] * len(ui_l))) if sr else [-100]
            ut_s_d = LocatSimiTerms(ut_l, t_set_s)

            for p_type in para_attr_l:
                p_t_d = {}
                for k, v in ut_s_d.items():
                    if not v and v != 0:
                        p_t_d[k] = None
                    else:
                        # v out of para range
                        try:
                            p_t_d[k] = getattr(p, p_type)[v]
                        except:
                            p_t_d[k] = None
                # para_dict[p_type] = p_t_d
                setattr(p, p_type, p_t_d)

        # for p_type in para_attr_l:
        #     p_t_l = [getattr(p, p_type).values() for p in para_data]
        #     p_t_l.reverse()
        #     para_dict[p_type] = p_t_l

        # TODO Reverse order out (tail to head)
        # return para_dict

    def RespSplicing(self, vm_cond: list, t_set: int) -> list:
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
        resp_info = {'data_val': False, 'mode_val': False}

        para_data = self.__ridrec.paras.copy()
        wave_data = self.__ridrec.waves.copy()

        if not wave_data or not para_data:
            return resp_select, resp_info
        else:
            resp_info['data_val'] = True

        vm_l = []  # save ventmode of paras
        val_t = lambda x, y: sum([True if i in x else False for i in y])

        for p in reversed(para_data):
            vm_s = list(reversed(p.st_mode))
            vm_l.extend(vm_s)

        ut_s = self.__UiIndexSplicing()  # UiDataIndex Splice

        p_str = 0
        p_end = LocatSimiTerms(ut_s, [t_set])[t_set]

        def GetWindowVMValidity(vm_list):
            vm_val = [val_t(v, vm_cond) > 0 for v in vm_list]
            vm_l_val = reduce(lambda x, y: x & y, vm_val)
            return vm_l_val

        if not self.__ridrec.op_t:
            vm_window = vm_l[slice(p_str, p_end)]
            vm_win_val = GetWindowVMValidity(vm_window)
        else:
            while p_end < len(vm_l):
                vm_window = vm_l[slice(p_str, p_end)]
                vm_win_val = GetWindowVMValidity(vm_window)
                if vm_win_val:
                    break
                else:
                    p_str += 1
                    p_end += 1

        def RespDataConcat(wave_s):
            resp_data = []
            v_still_t = 0
            for wave in reversed(wave_s):
                for resp in reversed(wave.resps):

                    if v_still_t > t_set:
                        break
                    elif not resp.val:
                        continue
                    else:
                        v_still_t += resp.wid
                        resp_data.append(resp)

            return resp_data

        if not vm_win_val:
            pass
        else:
            resp_info['mode_val'] = True
            resp_select = RespDataConcat(wave_data)
        # Positive order (range head to op-tail)
        return list(reversed(resp_select)), resp_info

    def ParaSplicing(self, para_attr_l: list, t_set: int = -1):
        """
        ababab
        """
        para_select = FromkeysReid(para_attr_l)

        para_data = self.__ridrec.paras.copy()

        if not para_data:
            return {}

        ut_s = self.__UiIndexSplicing()
        if t_set > 0:
            p_slice = slice(0, LocatSimiTerms(ut_s, [t_set])[t_set])
        else:
            p_slice = slice(0, len(ut_s))

        for p in reversed(para_data):
            for p_type in para_attr_l:
                p_list = list(reversed(getattr(p, p_type)))
                para_select[p_type].extend(p_list)

        for p_type in para_attr_l:
            p_in = para_select[p_type][p_slice]
            para_select[p_type] = list(reversed(p_in))

        ut_s = list(reversed(ut_s[p_slice]))
        ut_s = list(map(sub, ut_s, [ut_s[0]] * len(ut_s)))
        ut_s = list(map(mul, ut_s, [-1] * len(ut_s)))
        ut_s[0] = 0
        para_select['ind'] = ut_s

        return para_select