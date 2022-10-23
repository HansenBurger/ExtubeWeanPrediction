import math
import warnings
import numpy as np
import pandas as pd
# import entropy as ent
import EntropyHub as eh
from scipy import interpolate
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.stats import mannwhitneyu, normaltest, ttest_ind, spearmanr, pearsonr

warnings.filterwarnings('ignore')


class Basic():
    def __init__(self) -> None:
        self.round_i_0 = 4
        self.round_i_1 = 3


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
        self.__param_range = {
            'PIP': (10, 50),
            'PEEP': (0, 20),
            'RR': (0, 60),
            'V_T': (0, 2000),
            'VE': (0, 30),
            'MP': (0, 20)
        }

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

    def __ValRangeCheck(self, val_s: list, val_r: tuple) -> bool:
        '''
        Check if the index value is within the range
        '''
        val_s = [val_s] if not type(val_s) == list else val_s
        val_s = [-999 if math.isnan(v) else v for v in val_s]
        tri_l = [True if val_r[0] <= v <= val_r[1] else False for v in val_s]
        triger = True if not False in tri_l else False
        return triger

    def RespT(self, sample_rate: float) -> float:
        '''
        Count "resp still time (tot, in, ex)" per resp
        Unit: Second (s)
        '''
        tot_len = self.__end_ind - self.__pro_ind
        in_len = self.__min_ind - self.__pro_ind
        ex_len = self.__end_ind - self.__min_ind
        t_tot = tot_len * 1 / sample_rate
        t_in = in_len * 1 / sample_rate
        t_ex = ex_len * 1 / sample_rate
        in_ex_ratio = t_in / t_ex
        resp_t = {
            'wid': round(t_tot, self.round_i_0),
            't_i': round(t_in, self.round_i_0),
            't_e': round(t_ex, self.round_i_0),
            'i_e': round(in_ex_ratio, self.round_i_0)
        }
        return resp_t

    def PIP(self):
        '''
        Count "Peak Inspiratory Pressure" per resp
        Unit: cmH2O
        '''
        pip = self.__p_in[-1]  # max(self.__p_in)
        pip_val = self.__ValRangeCheck(pip, self.__param_range['PIP'])
        return round(pip, self.round_i_0), pip_val

    def PEEP(self):
        '''
        Count "Positive End-Expiratory Pressure" per resp
        Unit: cmH2O
        '''
        peep = self.__p_in[0]
        peep_val = self.__ValRangeCheck(peep, self.__param_range['PEEP'])
        return round(peep, self.round_i_0), peep_val

    def RR(self, sample_rate: float) -> list:
        '''
        Count "resp rate" per resp
        Unit: N/min (n/min)
        '''
        vent_len = self.__end_ind - self.__pro_ind
        RR = 60 / (vent_len * 1 / sample_rate)
        RR_val = self.__ValRangeCheck(RR, self.__param_range['RR'])
        return round(RR, self.round_i_0), RR_val

    def V_T(self) -> list:
        '''
        Count "tidal volume during expiratory" per resp
        Unit: ml (ml)
        '''
        v_in = self.__v_in
        v_ex = self.__v_ex
        v_t_i = v_in[-1]
        v_t_e = v_in[-1] + (v_ex[-1] if v_ex[-1] < 0 else -v_ex[-1])
        v_t = {
            'v_t_i': round(v_t_i, self.round_i_0),
            'v_t_e': round(v_t_e, self.round_i_0)
        }
        v_t_val = self.__ValRangeCheck([v_t_i, v_t_e],
                                       self.__param_range['V_T'])
        return v_t, v_t_val

    def VE(self, rr: float, v_t: float) -> list:
        '''
        Count "minute ventilation" per resp
        Unit: L/min (L/min)
        '''
        VE = rr * (v_t / 1000)
        val = self.__ValRangeCheck(VE, self.__param_range['VE'])
        return round(VE, self.round_i_0), val

    def RSBI(self, rr: float, v_t: float) -> float:
        '''
        Count "rapid shallow breathing index" per resp 
        Unit: Index (f/L)
        '''
        rsbi = rr / (v_t / 1000)
        return round(rsbi, self.round_i_0)

    def MP_Breath(self) -> list:
        '''
        Count work of the ventilator per breath
        mp_jb_d: MP d, Dynamic, J/breath
        mp_jb_t: MP t, Dynamic + Static, J/breath
        mp_jb_a: MP a, Overcome resistive power 
        mp_jb_b: MP b, Overcome elastic power
        Unit: J/breath (J/breath)
        '''
        p_in, v_in = self.__p_in, self.__v_in
        vp_rectangle = (p_in[-1] * v_in[-1]) / 1000
        peep_rectangle = (p_in[0] * v_in[-1]) / 1000
        inhal_points = abs(np.trapz(v_in, p_in)) / 1000

        mp_jb_t = vp_rectangle - inhal_points
        mp_jb_d = mp_jb_t - peep_rectangle
        mp_jb_b = ((p_in[-1] - p_in[0]) * v_in[-1]) / 2000
        mp_jb_a = mp_jb_d - mp_jb_b

        mp_jb_d *= 0.098
        mp_jb_t *= 0.098
        mp_jb_a *= 0.098
        mp_jb_b *= 0.098

        mp_jb_val = self.__ValRangeCheck([mp_jb_d, mp_jb_t],
                                         self.__param_range['MP'])
        mp_jb = {
            'mp_jb_d': round(mp_jb_d, self.round_i_0),
            'mp_jb_t': round(mp_jb_t, self.round_i_0),
            'mp_jb_a': round(mp_jb_a, self.round_i_0),
            'mp_jb_b': round(mp_jb_b, self.round_i_0)
        }
        return mp_jb, mp_jb_val

    def MP_Area(self, rr: float, v_t: float, wob: float) -> list:
        '''
        Count the work of the ventilator using the area method
        mp_jm_area: MP(J/min)
        mp_jl_area: MP(J/L)
        '''
        mp_jm_area = wob * rr
        mp_jl_area = wob / (v_t * 0.001)
        mp_area = {
            'mp_jm': round(mp_jm_area, self.round_i_0),
            'mp_jl': round(mp_jl_area, self.round_i_0)
        }
        mp_val = self.__ValRangeCheck([mp_jm_area, mp_jl_area],
                                      self.__param_range['MP'])
        return mp_area, mp_val


class VarCount(Basic):
    def __init__(self, ind_t: list = [], **kwargs) -> None:
        super().__init__()
        self.__arr_t = np.array(ind_t) if type(ind_t) != np.ndarray else ind_t
        self.__arr_0 = np.array(self.__arr_t[:len(self.__arr_t) - 1])
        self.__arr_1 = np.array(self.__arr_t[1:])


class TD(VarCount):
    def __init__(self, ind_t: list = [], **kwargs) -> None:
        super().__init__(ind_t, **kwargs)
        self.__ave = np.mean(self._VarCount__arr_t)
        self.__std = np.std(self._VarCount__arr_t)
        self.__cv = self.__std / self.__ave
        self.__med = np.median(self._VarCount__arr_t)
        self.__qua = np.quantile(self._VarCount__arr_t, 0.25)
        self.__tqua = np.quantile(self._VarCount__arr_t, 0.75)

    @property
    def ave(self):
        return self.__ave

    @property
    def std(self):
        return self.__std

    @property
    def cv(self):
        return self.__cv

    @property
    def med(self):
        return self.__med

    @property
    def qua(self):
        return self.__qua

    @property
    def tqua(self):
        return self.__tqua


class HRA(VarCount):
    def __init__(self, ind_t: list = [], **kwargs) -> None:
        super().__init__(ind_t, **kwargs)
        self.__pi = self.__PI()
        self.__gi = self.__GI()
        self.__si = self.__SI()

    @property
    def pi(self):
        return self.__pi

    @property
    def gi(self):
        return self.__gi

    @property
    def si(self):
        return self.__si

    def __PI(self) -> float:
        '''
        Count PI(the number of point below LI)
        '''
        a_0 = self._VarCount__arr_0.copy()
        a_1 = self._VarCount__arr_1.copy()

        num_upper, num_lower = 0, 0
        num_judge = lambda x, y: 1 if x < y else 0

        for i in range(len(a_0)):
            num_upper += num_judge(a_0[i], a_1[i])
            num_lower += num_judge(a_1[i], a_0[i])

        pi = 100 * num_lower / (num_upper + num_lower)
        return round(pi, self.round_i_0)

    def __GI(self) -> float:
        '''
        Count GI(the distance contributed by the points above LI)
        '''
        a_0 = self._VarCount__arr_0.copy()
        a_1 = self._VarCount__arr_1.copy()

        dis_above_sum, dis_below_sum = 0, 0
        dis_count = lambda x, y: (y - x) / np.sqrt(2) if x < y else 0

        for i in range(len(a_0)):
            dis_above_sum += dis_count(a_0[i], a_1[i])
            dis_below_sum += dis_count(a_1[i], a_0[i])
        gi = 100 * dis_above_sum / (dis_above_sum + dis_below_sum)
        return round(gi, self.round_i_0)

    def __SI(self) -> float:
        '''
        Count SI(the phase angle difference of the points above LI)
        '''
        a_0 = self._VarCount__arr_0.copy()
        a_1 = self._VarCount__arr_1.copy()

        theta_above_sum, theta_below_sum = 0, 0
        a_0, a_1 = np.abs(a_0), np.abs(a_1)  # make sure all positive
        theta_count = lambda x, y: np.degrees(np.arctan(y / x)
                                              ) - 45 if x < y else 0

        for i in range(len(a_0)):
            theta_above_sum += theta_count(a_0[i], a_1[i])
            theta_below_sum += theta_count(a_1[i], a_0[i])

        si = 100 * theta_above_sum / (theta_above_sum + theta_below_sum)
        return round(si, self.round_i_0)


class HRV(VarCount):
    def __init__(self, ind_t: list = [], **kwargs) -> None:
        super().__init__(ind_t, **kwargs)
        self.__sd1 = self.__SD1()
        self.__sd2 = self.__SD2()

    def sd1(self):
        return self.__sd1

    def sd2(self):
        return self.__sd2

    def __SD1(self) -> float:
        '''
        Count SD1(Short axis standard deviation)
        '''
        a_0 = self._VarCount__arr_0.copy()
        a_1 = self._VarCount__arr_1.copy()
        sd = np.std(a_1 - a_0) / np.sqrt(2)
        return round(sd, self.round_i_0)

    def __SD2(self) -> float:
        '''
        Count SD2(Long axis standard deviation)
        '''
        a_0 = self._VarCount__arr_0.copy()
        a_1 = self._VarCount__arr_1.copy()
        sd = np.std(a_1 + a_0) / np.sqrt(2)
        return round(sd, self.round_i_0)


class ENT(VarCount):
    def __init__(self, ind_t: list = [], **kwargs) -> None:
        super().__init__(ind_t, **kwargs)

    def __init__(self, ind_t: list = []) -> None:
        super().__init__(ind_t)
        self.__apen = eh.ApEn(self._VarCount__arr_t, m=1)
        self.__sampen = eh.SampEn(self._VarCount__arr_t, m=1)
        self.__fuzzen = eh.FuzzEn(self._VarCount__arr_t, m=1)

    def apen(self):
        return self.__apen

    def sampen(self):
        return self.__sampen

    def fuzzen(self):
        return self.__fuzzen


class PRSA(VarCount):
    def __init__(self, ind_t: list = [], **kwargs) -> None:
        super().__init__(ind_t, **kwargs)
        self.__prsa_mean = None
        self.__ac = self.__main('AC', **kwargs)
        self.__dc = self.__main('DC', **kwargs)

    @property
    def ac(self):
        return self.__ac

    @property
    def dc(self):
        return self.__dc

    @property
    def prsa_mean(self):
        return self.__prsa_mean

    def __SpaceGen(self, arr, fs) -> np.ndarray:
        ind_0, ind_1 = arr.min(), arr.max()
        ind_num = round((ind_1 - ind_0) * fs)
        arr_ = np.linspace(ind_0, ind_1, ind_num, endpoint=False)
        arr_ = np.round(arr_, decimals=2)
        return arr_

    def __Resample(self, ind_s: list, resample_rate: float) -> np.ndarray:
        arr_v = self._VarCount__arr_t.copy()
        arr_t = np.array(ind_s)
        arr_t = np.array([sum(arr_t[0:i + 1]) for i in range(len(arr_t))])

        f = interpolate.interp1d(arr_t, arr_v)
        arr_t_i = self.__SpaceGen(arr_t, resample_rate)
        arr_v_i = f(arr_t_i)
        return arr_v_i

    def __main(self, method_sub: str, ind_s: list, L: int, F: float, T: int,
               s: int) -> float:
        array_ = self.__Resample(ind_s, F)

        def WtJudge(val: float) -> float:
            '''
            The coefficients in wavelet analysis
            '''
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
                clip = slice(i - L, i + L)
                anchor_s.append(array_[clip].tolist())

        arr_prsa = np.array([np.mean(i) for i in np.array(anchor_s).T])
        arr_axis = np.linspace(-L, L, 2 * L, endpoint=False)
        df_prsa = pd.DataFrame({'axis': arr_axis, 'prsa': arr_prsa})

        self.__prsa_mean = df_prsa

        arr_axis_s = np.linspace(-s, s, 2 * s, endpoint=False)
        arr_prsa_s = df_prsa[df_prsa.axis.isin(arr_axis_s)].prsa
        arr_para_s = np.array([WtJudge(i / s) for i in arr_axis_s])
        prsa_ana = np.sum(arr_prsa_s * arr_para_s / s)

        return round(prsa_ana, self.round_i_0)





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

    def AucAssess(self) -> list:
        fpr, tpr, _ = roc_curve(self.__true_a, self.__pred_a)
        auc = round(roc_auc_score(self.__true_a, self.__pred_a),
                    self.round_i_1)
        return auc, fpr, tpr

    def ChiSquareAssess(self):
        chi_square = round(chi2(self.__true_a, self.__pred_a), self.round_i_1)
        return chi_square

    def MutInfoAssess(self):
        mut_info = round(mutual_info_classif(self.__true_a, self.__pred_a),
                         self.round_i_1)
        return mut_info

    def PValueAssess(self, alpha: float = 0.05, cate: str = 'continuous'):
        # Perform type: binary, continuous

        pos_, neg_ = self.__PosNegSep()
        _, p_1 = normaltest(pos_) if pos_.shape[0] >= 8 else None, 0
        _, p_2 = normaltest(neg_) if neg_.shape[0] >= 8 else None, 0

        def GetDist_0(arr):
            ave = round(np.mean(arr), self.round_i_1)
            std = round(np.std(arr), self.round_i_1)
            dist_ = '{0} +- {1}'.format(ave, std)
            return dist_

        def GetDist_1(arr):
            med = round(np.median(arr), self.round_i_1)
            qua = round(np.percentile(arr, 25), self.round_i_1)
            tqua = round(np.percentile(arr, 75), self.round_i_1)
            dist_ = '{0} ({1}, {2})'.format(med, qua, tqua)
            return dist_

        def GetDist_2(arr):
            len_0 = np.sum(arr == 0)
            len_1 = np.sum(arr == 1)
            ratio = round(len_0 / (len_0 + len_1) * 100, 2)
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


class Correlations(Basic):
    def __init__(self) -> None:
        super().__init__()

    def PearsonIndex(self):
        pass


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