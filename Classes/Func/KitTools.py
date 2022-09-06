import json
import numpy as np
import pandas as pd
from time import time
from pathlib import Path, PurePath
from functools import wraps
from datetime import datetime, timedelta
from copy import deepcopy

# config_file = r'C:\Users\HY_Burger\Desktop\Project\ExtubeWeanPrediction\config.json'


def measure(func):
    '''
    mainfunc: count time consuming of the function
    '''
    @wraps(func)
    def _time_it(*args, **kwargs):
        print(func.__name__, 'starts running...')
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            if end_ < 0:
                end_ = str(0) + ' ms'
            elif end_ > 10000:
                end_ = str(timedelta(seconds=end_ / 1000))
            else:
                end_ = str(end_) + ' ms'
            print("Total execution time: {0}\n".format(end_))

    return _time_it


def GetObjectDict(obj):
    keys = []
    vals = obj.__dict__.values()
    for i in obj.__dict__.keys():
        if not type(obj).__name__ in i:
            key_ = i
        else:
            key_ = i.split('_' + type(obj).__name__ + '__')[1]
        keys.append(key_)
    dict_ = dict(zip(keys, vals))
    return dict_


def FromkeysReid(dict_name, type_=[]):
    '''
    mainfunc: creat a dict by list and re_id each variable
    '''

    dict_ = dict.fromkeys(dict_name)
    for i in dict_name:
        tmp = type_
        dict_[i] = deepcopy(tmp)

    return dict_


def PathVerify(loc: PurePath) -> PurePath:
    return Path(loc) if not isinstance(loc, PurePath) else loc


def LocatSimiTerms(l_main: list, l_depend: list):
    index_dict = {}
    l_main = np.array(l_main)
    for i in l_depend:
        if i > l_main.max():
            index_dict[i] = None
        elif i < l_main.min():
            index_dict[i] = None
        else:
            clo_value = min(l_main, key=lambda x: abs(x - i))
            clo_index = np.where(l_main == clo_value)[0][0]
            index_dict[i] = clo_index

    return index_dict


def ConfigRead(cate: str,
               name: str = '',
               file: Path = Path('config.json')) -> str:

    if not file.is_file():
        print('Json File Not Exist !')
        return None
    else:
        with open(str(file), encoding="utf-8") as f:
            data = json.load(f)

        if not name:
            return data[cate]
        else:
            return data[cate][name]


def DLToLD(DL: dict) -> list:
    DL = {k: [v] if type(v) != list else v for k, v in DL.items()}
    max_len = max([len(i) for i in DL.values()])
    DL = {k: v * max_len if len(v) < max_len else v for k, v in DL.items()}
    LD = [dict(zip(DL, t)) for t in zip(*DL.values())]
    return LD


def TimeShift(df: pd.DataFrame, column_names: list) -> None:

    date_format = ['%Y/%m/%d %X', '%Y-%m-%d %X', '%Y-%m-%d %H:%M']

    for i in df.columns:

        if i in column_names:

            for fmt in date_format:
                try:
                    df[i] = pd.to_datetime(df[i], format=fmt)
                    break
                except ValueError:
                    if fmt == date_format[-1]:
                        print(
                            'The "{0}" of table cannot find "ANY" corresponding time format'
                            .format(i))


def SaveGen(p_loc: Path, fold_n: str) -> Path:
    now = datetime.now()
    p_loc = Path(p_loc) if not isinstance(p_loc, PurePath) else p_loc
    fold_pre = '{0}{1}{2}_{3}_'.format(now.year,
                                       str(now.month).rjust(2, '0'),
                                       str(now.day).rjust(2, '0'),
                                       str(now.hour).rjust(2, '0'))
    save_loc = p_loc / (fold_pre + fold_n)
    save_loc.mkdir(parents=True, exist_ok=True)
    return save_loc