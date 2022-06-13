import math
import warnings
import numpy as np
import pandas as pd
import entropy as ent
import EntropyHub as eh
from scipy import interpolate
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy.stats import normaltest, ttest_ind

warnings.filterwarnings('ignore')


class Basic():
    def __init__(self) -> None:
        pass


class IndCalculation(Basic):
    def __init__(self, p_start, p_end):
        '''
        Calculate the value of each indicator for each breath
        p_start: Point of the resp start
        p_end: Point of the resp end
        '''
        super().__init__()
        self.__pro_ind = p_start
        self.__end_ind = p_end
        self.__min_ind = 0  # point of inhalation and exhalation transition
        self.__p_in = np.array([])  # Inhalation P
        self.__p_ex = np.array([])  # Exhalation P
        self.__v_in = np.array([])  # Inhalation V
        self.__v_ex = np.array([])  # Exhalation V

        self.__valid_tag = True  # Validity of current resp

    @property
    def valid_tag(self):
        return self.__valid_tag

    def __GapDetection(self) -> bool:
        '''
        Resp width detection
        '''
        min_gap = 50
        max_gap = 450
        gap = abs(self.__end_ind - self.__pro_ind)

        if gap < min_gap or gap > max_gap or gap == 0:
            return False
        else:
            return True

    def __LineDetection(self, array_: np.array) -> bool:
        '''
        Whether the respiratory waveform is straight line detection
        '''
        if np.all(array_ == array_[0]):
            return False
        else:
            return True

    def __SwitchPoint(self, array_: np.array) -> int:
        '''
        Search point of inspiratory and expiratory transition
        '''
        interval = 15
        ind_array = np.where(array_[interval:] < 0)[0]
        try:
            ind = ind_array[0] + interval
        except:
            ind = None
        return ind

    def __LenMatch(self, array_1: np.array, array_2: np.array) -> None:
        '''
        Weather two array length equal detection
        '''
        len_1 = array_1.shape[0]
        len_2 = array_2.shape[0]

        if len_1 != len_2:
            self.__valid_tag = False

    def __ArrayCertify(self, list_: list) -> None:
        '''
        Weather array empty detection
        '''
        list_ = [list_] if type(list_) == np.array else list_

        for array in list_:
            if not np.any(array):
                self.__valid_tag = False
                return

    def __OutliersDel(self, array_1: np.array, array_2: np.array) -> None:
        '''
        Delet the outliers in p_in&v_in or p_ex|v_ex
        '''
        def GetOutliers(arr_: np.array, max_d: int):
            dis_mean = arr_ - np.mean(arr_)
            outliers = dis_mean > max_d * np.std(arr_)
            return outliers

        max_deviation = 4

        arr_1_outs = GetOutliers(array_1, max_deviation)
        arr_2_outs = GetOutliers(array_2, max_deviation)

        arrs_not_outs = ~(arr_1_outs | arr_2_outs)

        array_1 = array_1[arrs_not_outs]
        array_2 = array_2[arrs_not_outs]

        return array_1, array_2

    def ValidityCheck(self, s_F: list, s_V: list, s_P: list) -> None:
        '''
        Checking the validity of the respiratory waveform and 
        searching for inspiratory and expiratory switching points 
        in preparation for index calculations
        s_F: Resp flow sequence
        s_V: Resp volumn sequence
        s_P Resp pressure sequence
        '''
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

                    if self.__valid_tag:
                        self.__p_in, self.__v_in = self.__OutliersDel(
                            self.__p_in, self.__v_in)
                        self.__p_ex, self.__v_ex = self.__OutliersDel(
                            self.__p_ex, self.__v_ex)

                    if self.__v_in[-1] == 0:
                        self.__valid_tag = False

                else:
                    self.__valid_tag = False
            else:
                self.__valid_tag = False
        else:
            self.__valid_tag = False

    def __ValRangeCheck(self, vals: list, val_range: range) -> bool:
        '''
        Check if the index value is within the range
        '''
        vals = [vals] if not type(vals) == list else vals
        vals = [-999 if math.isnan(v) else v for v in vals]
        tri_l = [True if round(v) in val_range else False for v in vals]
        triger = True if not False in tri_l else False
        return triger

    def RespT(self, sample_rate: float) -> float:
        '''
        Count "resp still time" per resp
        Unit: Second (s)
        '''
        vent_len = self.__end_ind - self.__pro_ind
        resp_t = vent_len * 1 / sample_rate
        return round(resp_t, 2)

    def RR(self, sample_rate: float) -> list:
        '''
        Count "resp rate" per resp
        Unit: N/min (n/min)
        '''
        vent_len = self.__end_ind - self.__pro_ind
        RR = 60 / (vent_len * 1 / sample_rate)
        RR_val = self.__ValRangeCheck(RR, range(0, 60))
        return round(RR, 2), RR_val

    def V_t(self) -> list:
        '''
        Count "tidal volume during expiratory" per resp
        Unit: ml (ml)
        '''
        v_in = self.__v_in
        v_ex = self.__v_ex
        v_t_i = v_in[-1]
        v_t_e = v_in[-1] + (v_ex[-1] if v_ex[-1] < 0 else -v_ex[-1])
        v_t = {'v_t_i': round(v_t_i, 2), 'v_t_e': round(v_t_e, 2)}
        v_t_val = self.__ValRangeCheck([v_t_i, v_t_e], range(0, 2000))
        return v_t, v_t_val

    def WOB(self) -> list:
        '''
        Count "work of breath" in multiple modes per resp
        wob: WOB, Dynamic
        wob_f: WOB Full, Dynamic + Static
        wob_a: WOB A, Resistive WOB in
        wob_b: WOB B, Elastic WOB in
        Unit: J/L (J/L)
        '''
        p_in, v_in = self.__p_in, self.__v_in
        vp_rectangle = (p_in[-1] * v_in[-1]) / 1000
        peep_rectangle = (p_in[0] * v_in[-1]) / 1000
        inhal_points = abs(np.trapz(v_in, p_in)) / 1000

        wob_full = vp_rectangle - inhal_points
        wob = wob_full - peep_rectangle
        wob_b = ((p_in[-1] - p_in[0]) * v_in[-1]) / 2000
        wob_a = wob - wob_b

        wob_val = self.__ValRangeCheck([wob, wob_full], range(0, 20))
        wob_ = {
            'wob': round(wob, 2),
            'wob_f': round(wob_full, 2),
            'wob_a': round(wob_a, 2),
            'wob_b': round(wob_b, 2)
        }
        return wob_, wob_val

    def VE(self, rr: float, v_t: float) -> list:
        '''
        Count "minute ventilation" per resp
        Unit: L/min (L/min)
        '''
        VE = rr * (v_t / 1000)
        val = self.__ValRangeCheck(VE, range(0, 30))
        return round(VE, 2), val

    def RSBI(self, rr: float, v_t: float) -> float:
        '''
        Count "rapid shallow breathing index" per resp 
        Unit: Index (f/L)
        '''
        rsbi = rr / (v_t / 1000)
        return round(rsbi, 2)

    def MP_Area(self, rr: float, v_t: float, wob: float) -> list:
        '''
        Count the work of the ventilator using the area method
        mp_jm_area: MP(Jm)
        mp_jl_area: MP(JL)
        '''
        mp_jm_area = 0.098 * rr * wob
        mp_jl_area = 0.098 * wob / (v_t * 0.001)
        mp_area = {
            'mp_jm': round(mp_jm_area, 2),
            'mp_jl': round(mp_jl_area, 2)
        }
        mp_val = self.__ValRangeCheck([mp_jm_area, mp_jl_area], range(0, 20))
        return mp_area, mp_val


class VarAnalysis(Basic):
    def __init__(self, ind_t: list = []):
        super().__init__()
        self.__arr_t = np.array(ind_t)

    def TimeSeries(self, ind_s: list, method_sub: str) -> float:
        array_ = np.array(ind_s)
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

    def __Panglais(self, list_: list) -> np.array:
        a_0 = np.array(list_[:len(list_) - 1])
        a_1 = np.array(list_[1:])
        return a_0, a_1

    def __PI(self, a_0: np.ndarray, a_1: np.ndarray) -> float:
        '''
        Count PI(the point below LI)
        a_0: Array of IND_i
        a_1: Array of IND_i+1
        '''
        n = 0
        m = 0
        for i in range(len(a_0)):
            n += 1 if a_0[i] < a_1[i] else 0
            m += 1 if a_0[i] > a_1[i] else 0
        pi = 100 * m / (n + m)
        return round(pi, 2)

    def __GI(self, a_0: np.ndarray, a_1: np.ndarray) -> float:
        '''
        Count GI(the distance contributed by the points above LI)
        a_0: Array of IND_i
        a_1: Array of IND_i+1
        '''
        d1 = 0
        d2 = 0
        for i in range(len(a_0)):
            d1 += (a_1[i] - a_0[i]) / np.sqrt(2) if a_0[i] < a_1[i] else 0
            d2 += (a_0[i] - a_1[i]) / np.sqrt(2) if a_0[i] > a_1[i] else 0
        gi = 100 * d1 / (d1 + d2)
        return round(gi, 2)

    def __SI(self, a_0: np.ndarray, a_1: np.ndarray) -> float:
        '''
        Count SI(the phase angle difference of the points above LI)
        a_0: Array of IND_i
        a_1: Array of IND_i+1
        '''
        theta_1 = 0
        theta_2 = 0
        for i in range(len(a_0)):
            theta_1 += (np.degrees(np.arctan(a_1[i] / a_0[i])) -
                        45) if a_0[i] < a_1[i] else 0
            theta_2 += (-np.degrees(np.arctan(a_1[i] / a_0[i])) +
                        45) if a_0[i] > a_1[i] else 0
        si = 100 * theta_1 / (theta_1 + theta_2)
        return round(si, 2)

    def HRA(self, ind_s: list, method_sub: str, **kwargs) -> float:
        array_0, array_1 = self.__Panglais(ind_s)

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

    def __SD1(self, a_0, a_1):
        '''
        Count SD1()
        a_0: Array of IND_i
        a_1: Array of IND_i+1
        '''
        sd = np.std(a_1 - a_0) / np.sqrt(2)
        return round(sd, 2)

    def __SD2(self, a_0, a_1):
        '''
        Count SD2()
        a_0: Array of IND_i
        a_1: Array of IND_i+1
        '''
        sd = np.std(a_1 + a_0) / np.sqrt(2)
        return round(sd, 2)

    def HRV(self, ind_s: list, method_sub: str, **kwargs) -> float:
        array_0, array_1 = self.__Panglais(ind_s)

        if method_sub == 'SD1':
            result_ = self.__SD1(array_0, array_1)
        elif method_sub == 'SD2':
            result_ = self.__SD2(array_0, array_1)
        else:
            result_ = -999
            print('No match method')

        return result_

    def Entropy(self, ind_s: list, method_sub: str, **kwargs) -> float:
        array_ = np.array(ind_s)
        if method_sub == 'AppEn':
            result_, _ = eh.ApEn(array_, m=1)
        elif method_sub == 'SampEn':
            result_, _, _ = eh.SampEn(array_, m=1)
        elif method_sub == 'FuzzEn':
            result_, _, _ = eh.FuzzEn(array_, m=1)
        else:
            result_ = -999
            print('No match method')

        return round(result_[-1], 2)

    def __SpaceGen(self, arr, fs):
        ind_0 = arr.min()
        ind_1 = arr.max()
        ind_num = round((ind_1 - ind_0) * fs)
        arr_ = np.linspace(ind_0, ind_1, ind_num, endpoint=False)
        arr_ = np.round(arr_, decimals=2)
        return arr_

    def __Resample(self, ind_s: list, resample_rate: float) -> np.ndarray:
        arr_v = np.array(ind_s)
        arr_t = self.__arr_t.copy()
        arr_t = np.array([sum(arr_t[0:i + 1]) for i in range(len(arr_t))])

        f = interpolate.interp1d(arr_t, arr_v)
        arr_t_i = self.__SpaceGen(arr_t, resample_rate)
        arr_v_i = f(arr_t_i)
        return arr_v_i

    def PRSA(self, ind_s: list, method_sub: str, L: int, F: float, T: int,
             s: int) -> float:
        array_ = self.__Resample(ind_s, F)

        def WtJudge(val: float) -> float:
            if val >= -1 and val < 0:
                para = -1 / 2
            elif val >= 0 and val < 1:
                para = 1 / 2
            else:
                para = 0
            return para

        anchor_s = []

        if method_sub == 'AC':
            anchor_set = lambda x, y: True if np.mean(x[y:y + T]) > np.mean(x[
                y - T:y]) else False
        elif method_sub == 'DC':
            anchor_set = lambda x, y: True if np.mean(x[y:y + T]) < np.mean(x[
                y - T:y]) else False
        else:
            print('No match method')
            return

        for i in range(L, len(array_) - L):
            if not anchor_set(array_, i):
                pass
            else:
                clip = slice(i - L, i + L + 1)
                anchor_s.append(array_[clip].tolist())

        arr_prsa = np.array([np.mean(i) for i in np.array(anchor_s).T])
        arr_axis = np.linspace(-L, L + 1, 2 * L + 1, endpoint=False)
        df = pd.DataFrame({'axis': arr_axis, 'prsa': arr_prsa})

        arr_axis_s = np.linspace(-s, s, 2 * s, endpoint=False)
        arr_prsa_s = df[df.axis.isin(arr_axis_s)].prsa
        arr_para_s = np.array([WtJudge(i / s) for i in arr_axis_s])
        prsa_ana = np.sum(arr_prsa_s * arr_para_s / s)

        return round(prsa_ana, 4)


class PerfomAssess(Basic):
    def __init__(
        self,
        ture_l: list,
        pred_l: list,
    ) -> None:
        super().__init__()
        self.__true_a: np.ndarray = np.array(ture_l)
        self.__pred_a: np.ndarray = np.array(pred_l)

    def __PosNegSep(self) -> list:
        df = pd.DataFrame({'true': self.__true_a, 'pred': self.__pred_a})
        pos_arr = np.array(df[df.true == 1].pred)
        neg_arr = np.array(df[df.true == 0].pred)
        return pos_arr, neg_arr

    def AucAssess(self, v_ran: int = 3) -> list:
        fpr, tpr, _ = roc_curve(self.__true_a, self.__pred_a)
        auc = round(roc_auc_score(self.__true_a, self.__pred_a), v_ran)
        return auc, fpr, tpr

    def PAssess(self, alpha: float = 0.05, cate: str = 'continuous'):
        # Perform type: binary, continuous

        pos_, neg_ = self.__PosNegSep()
        _, p_1 = normaltest(pos_) if pos_.shape[0] >= 8 else None, 0
        _, p_2 = normaltest(neg_) if neg_.shape[0] >= 8 else None, 0

        def GetDist_0(arr):
            ave = round(np.mean(arr), 3)
            std = round(np.std(arr), 3)
            dist_ = '{0} +- {1}'.format(ave, std)
            return dist_

        def GetDist_1(arr):
            med = round(np.median(arr), 3)
            qua = round(np.percentile(arr, 25), 3)
            tqua = round(np.percentile(arr, 75), 3)
            dist_ = '{0} ({1}, {2})'.format(med, qua, tqua)
            return dist_

        def GetDist_2(arr):
            len_0 = np.sum(arr == 0)
            len_1 = np.sum(arr == 1)
            ratio = round(len_0 / len(arr), 3)
            dist_ = '{0} ({1})'.format(len_0, ratio)
            return dist_

        if p_1 > alpha and p_2 > alpha:
            len_euqal = len(pos_) == len(neg_)
            _, p = ttest_ind(pos_, neg_, equal_var=len_euqal)
            if cate == 'continuous':
                rs_pos = GetDist_0(pos_)
                rs_neg = GetDist_0(neg_)

            elif cate == 'binary':
                rs_pos = GetDist_2(pos_)
                rs_neg = GetDist_2(neg_)
        else:
            _, p = mannwhitneyu(pos_, neg_)
            if cate == 'continuous':
                rs_pos = GetDist_1(pos_)
                rs_neg = GetDist_1(neg_)

            elif cate == 'binary':
                rs_pos = GetDist_2(pos_)
                rs_neg = GetDist_2(neg_)

        p = round(p, 4) if not p < 0.0001 else 0.0001
        return p, rs_pos, rs_neg


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
    def __init__(self, time_array, target_array, range_=slice(None)):
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
        ind_num = round((ind_1 - ind_0) * fs)
        array = np.linspace(ind_0, ind_1, ind_num, endpoint=False)
        array = np.around(array, decimals=2)
        return array

    def InitTimeSeries(self):
        self.__LenVertify()
        df = pd.DataFrame({'time': self.__time_a, 'value': self.__target_a})
        self.__df = df

    def Resampling(self, resample_rate):
        self.__LenVertify()
        f = interpolate.interp1d(self.__time_a, self.__target_a)
        array_x = self.__SpaceGen(resample_rate)
        array_y = f(array_x)
        df = pd.DataFrame({'time': array_x, 'value': array_y})
        self.__df = df