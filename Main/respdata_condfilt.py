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

from Classes.Func.KitTools import measure


@measure
def main():
    pass


if __name__ == '__main__':
    main()