import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class Basic():
    def __init__(self) -> None:
        pass


class IndCalculation(Basic):
    def __init__(self, p_start, p_end):
        super().__init__()
        self.__pro_ind = p_start
        self.__end_ind = p_end
        self.__min_ind = 0
        self.__p_in = np.array([])
        self.__p_ex = np.array([])
        self.__v_in = np.array([])
        self.__v_ex = np.array([])

        self.__valid_tag = True

    @property
    def valid_tag(self):
        return self.__valid_tag

    def __GapDetection(self):
        min_gap = 50
        max_gap = 450
        gap = abs(self.__end_ind - self.__pro_ind)

        if gap < min_gap or gap > max_gap or gap == 0:
            return False
        else:
            return True

    def __LineDetection(self, array_):

        if np.all(array_ == array_[0]):
            return False
        else:
            return True

    def __SwitchPoint(self, array_):
        interval = 15
        ind_array = np.where(array_[interval:] < 0)[0]
        try:
            ind = ind_array[0] + interval
        except:
            ind = None
        return ind

    def __LenMatch(self, array_1, array_2):
        len_1 = array_1.shape[0]
        len_2 = array_2.shape[0]

        if len_1 != len_2:
            self.__valid_tag = False

    def __ArrayCertify(self, list_):
        list_ = [list_] if type(list_) == np.array else list_

        for array in list_:
            if not np.any(array):
                self.__valid_tag = False
                return

    def ValidityCheck(self, s_F, s_V, s_P):
        p_i, e_i, m_i = self.__pro_ind, self.__end_ind, self.__min_ind

        if self.__GapDetection():
            f_wave = np.array(s_F[p_i:e_i])
            if self.__LineDetection(f_wave) and self.__SwitchPoint(f_wave):
                m_i = self.__SwitchPoint(f_wave) + p_i
                if m_i != p_i:

                    self.__min_ind = m_i
                    self.__p_in = np.array(s_P[p_i:m_i])
                    self.__p_ex = np.array(s_P[m_i:e_i])
                    self.__v_in = np.array(s_V[p_i:m_i])
                    self.__v_ex = np.array(s_V[m_i:e_i])

                    self.__ArrayCertify(
                        [self.__p_in, self.__p_ex, self.__v_in, self.__v_ex])
                    self.__LenMatch(self.__p_in, self.__v_in)
                    self.__LenMatch(self.__p_ex, self.__v_ex)

                    if self.__v_in[-1] == 0:
                        self.__valid_tag = False

                else:
                    self.__valid_tag = False
            else:
                self.__valid_tag = False
        else:
            self.__valid_tag = False

    def RespT(self, sample_rate):
        vent_len = self.__end_ind - self.__pro_ind
        resp_t = vent_len * 1 / sample_rate
        return round(resp_t, 2)

    def RR(self, sample_rate):
        vent_len = self.__end_ind - self.__pro_ind
        RR = 60 / (vent_len * 1 / sample_rate)
        return round(RR, 2)

    def V_t_i(self):
        v_t_i = self.__v_in[-1]
        return round(v_t_i, 2)

    def V_t_e(self):
        v_in, v_ex = self.__v_in, self.__v_ex
        v_t_e = v_in[-1] + (v_ex[-1] if v_ex[-1] < 0 else -v_ex[-1])
        return round(v_t_e, 2)

    def WOB(self):
        p_in, v_in = self.__p_in, self.__v_in
        vp_rectangle = (p_in[-1] * v_in[-1]) / 1000
        peep_rectangle = (p_in[0] * v_in[-1]) / 1000
        inhal_points = abs(np.trapz(v_in, p_in)) / 1000

        wob_full = vp_rectangle - inhal_points
        wob = wob_full - peep_rectangle
        wob_b = ((p_in[-1] - p_in[0]) * v_in[-1]) / 2000
        wob_a = wob - wob_b

        return {
            'wob': round(wob, 2),
            'wob_f': round(wob_full, 2),
            'wob_a': round(wob_a, 2),
            'wob_b': round(wob_b, 2)
        }

    def VE(self, rr, v_t):
        VE = rr * (v_t / 1000)
        return round(VE, 2)

    def RSBI(self, rr, v_t):
        rsbi = rr / (v_t / 1000)
        return round(rsbi, 2)

    def MP_Area(self, rr, v_t, wob):
        mp_jm_area = 0.098 * rr * wob
        mp_jl_area = 0.098 * wob / (v_t * 0.001)
        return {'mp_jm': round(mp_jm_area, 2), 'mp_jl': round(mp_jl_area, 2)}


class VarAnalysis(Basic):
    def __init__(self):
        super().__init__()

    def __PI(self, a_0, a_1):
        n = 0
        m = 0
        for i in range(len(a_0)):
            n += 1 if a_0[i] < a_1[i] else 0
            m += 1 if a_0[i] > a_1[i] else 0
        pi = 100 * m / (n + m)
        return round(pi, 2)

    def __GI(self, a_0, a_1):
        d1 = 0
        d2 = 0
        for i in range(len(a_0)):
            d1 += (a_1[i] - a_0[i]) / np.sqrt(2) if a_0[i] < a_1[i] else 0
            d2 += (a_0[i] - a_1[i]) / np.sqrt(2) if a_0[i] > a_1[i] else 0
        gi = 100 * d1 / (d1 + d2)
        return round(gi, 2)

    def __SI(self, a_0, a_1):
        theta_1 = 0
        theta_2 = 0
        for i in range(len(a_0)):
            theta_1 += (np.degrees(np.arctan(a_1[i] / a_0[i])) -
                        45) if a_0[i] < a_1[i] else 0
            theta_2 += (-np.degrees(np.arctan(a_1[i] / a_0[i])) +
                        45) if a_0[i] > a_1[i] else 0
        si = 100 * theta_1 / (theta_1 + theta_2)
        return round(si, 2)

    def __SD1(self, a_0, a_1):
        sd = np.std(a_1 - a_0) / np.sqrt(2)
        return round(sd, 2)

    def __SD2(self, a_0, a_1):
        sd = np.std(a_1 + a_0) / np.sqrt(2)
        return round(sd, 2)

    def __Panglais(self, list_):
        a_0 = np.array(list_[:len(list_) - 1])
        a_1 = np.array(list_[1:])
        return a_0, a_1

    def TimeSeries(self, list_, method_sub):
        array_ = np.array(list_)

        if method_sub == 'AVE':
            result_ = np.mean(array_)
        elif method_sub == 'STD':
            result_ = np.std(array_)
        elif method_sub == 'CV':
            result_ = np.std(array_) / np.mean(array_)
        elif method_sub == 'MED':
            result_ = np.median(array_)
        elif method_sub == 'QUA':
            result_ = np.quantile(array_, 0.25)
        elif method_sub == 'TQUA':
            result_ = np.quantile(array_, 0.75)
        else:
            result_ = -999
            print('No match method')

        return round(result_, 2)

    def HRA(self, list_, method_sub):
        array_0, array_1 = self.__Panglais(list_)

        if method_sub == 'PI':
            result_ = self.__PI(array_0, array_1)
        elif method_sub == 'GI':
            result_ = self.__GI(array_0, array_1)
        elif method_sub == 'SI':
            result_ = self.__SI(array_0, array_1)
        else:
            result_ = -999
            print('No match method')

        return result_

    def HRV(self, list_, method_sub):
        array_0, array_1 = self.__Panglais(list_)

        if method_sub == 'SD1':
            result_ = self.__SD1(array_0, array_1)
        elif method_sub == 'SD2':
            result_ = self.__SD2(array_0, array_1)
        else:
            result_ = -999
            print('No match method')

        return result_


class SenSpecCounter(Basic):
    def __init__(self, value_depend, value_array):
        super().__init__()
        self.__val_a = value_array.copy()
        self.__val_d = value_depend

    def __SensSpec(self, true_a, pred_a):
        confusion = confusion_matrix(true_a, pred_a)
        TP = confusion[1, 1]  # true positive
        TN = confusion[0, 0]  # true negatives
        FP = confusion[0, 1]  # false positives
        FN = confusion[1, 0]  # false negatives
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        return round(sens, 2), round(spec, 2)

    def CutEndPos(self, true_a, d_type):
        pred_a = (self.__val_a > self.__val_d).astype(d_type)
        sens, spec = self.__SensSpec(true_a, pred_a)
        return {'sep': self.__val_d, 'sens': sens, 'spec': spec}

    def CutEndNeg(self, true_a, d_type):
        pred_a = (self.__val_a < self.__val_d).astype(d_type)
        sens, spec = self.__SensSpec(true_a, pred_a)
        return {'sep': self.__val_d, 'sens': sens, 'spec': spec}


class FreqPreMethod():
    def __init__(self, time_array, target_array, range_):
        self.__time_a = time_array[range_]
        self.__target_a = target_array[range_]
        self.__df = None

    @property
    def df(self):
        return self.__df

    def __LenVertify(self):
        if not len(self.__time_a) == len(self.__target_a):
            print('Length Mismatch !')
            return

    def __SpaceGen(self, fs):
        ind_0 = self.__time_a.min()
        ind_1 = self.__time_a.max()
        ind_num = (ind_1 - ind_0) * fs
        array = np.linspace(ind_0, ind_1, ind_num, endpoint=False)
        array = np.around(array, decimals=2)
        return array

    def InitTimeSeries(self):
        self.__LenVertify()
        df = pd.DataFrame({'time': self.__time_a, 'value': self.__target_a})
        self.__df = df

    def InterpValue(self, interp_rate):
        self.__LenVertify()
        array_x = self.__SpaceGen(interp_rate)
        array_y = np.interp(array_x, self.__time_a, self.__target_a)
        df = pd.DataFrame({'time': array_x, 'value': array_y})
        self.__df = df

    def Resampling(self, resample_rate):
        self.__LenVertify()
        df = self.__df.copy()
        array_x = self.__SpaceGen(resample_rate)
        array_y = np.array(
            [df.loc[df['time'] == i]['value'].item() for i in array_x])
        df_ = pd.DataFrame({'time': array_x, 'value': array_y})
        self.__df = df_