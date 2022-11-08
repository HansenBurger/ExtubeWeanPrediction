import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

pid = [6913426, 3837466, 6640624, 6003532]
'''
Process workflow
1. Copy data from server
    1.1 Source: zres_param, pid_required
    1.2 Condition: Unique Rid, total vent time ≤ 2 days (manually)
    1.3 Result: table like Extube_PSV
2. Calculate the breathing variability in total vent time
    2.1 Condition: mode -- PCV
    2.2 Calculation: 
        (1) Values of respiratory parameters throughout ventilation
        (2) Variability results per hour, per 30 min?(changable)
    2.3 Storage Intermediate:
3. Display results
    3.1 resp param trend plot(line plot total, scatter plot per hour)
    3.2 variability trend plot
'''