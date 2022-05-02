import numpy as np

paper_0 = '''
DHMV, Mean airway pressure, VT, VE, PIP, PO2, PaCO2, Arterial PH, BUN, Hemoglobin, 
Blood glucose, APTT, Cl-, K+, Platelet,  Ca+, WBC, Na+, Lactic acid, Prothrombin time, 
INR, Base excess, HR, RR, Temperature, Weight, Age, Diastolic blood pressure, 
Systolic blood pressure, Percutaneous oxygen saturation, Height ,Arrhythmia, Gender, 
Weight loss, Stroke, Chronic cardiac insufficiency, Coagulopathy, COPD, 
Electrolyte and fluid disorders, Hypertension, DU, hypothyroidism, Peripheral vascular, 
Drug abuse psychoses, Chronic kidney disease, Other neurological, Psychoses, 
Metastatic tumors, Depression, Alcohol abuse, Valcular disease, Liver disease, Obesity, 
Rheumatoid arthritis, Solid tumor, Lymphoma, DC, Pulmonary heart disease, 
deficiency anemias, blood loss anemias, Aids, Peptic ulcer, GCS, SOFA, 
Sedation day, Vasopressor'''

paper_1 = '''
VE, PIP, RSBI, PaO2, PaCO2, Arterial PH, RR, Gender, GCS'''

paper_2 = '''
VT, VE, Mean airway pressure, PEEP, PSV level(cmH2O), SpO2, PaCO2, Arterial PH, 
CVP, Urine output, Crystalloid amount, HR, RR, Temperature, Mean arterial pressure(MAP), 
Age, BMI, Stroke, MV duration, SBT success time, Antibiotic types'''

paper_3 = '''
Ventilation mode, VT, PIP, PPLAT, RSBI, SpFiO2, ROX, RASS, GCS, Total given dose, 
Total cumulative dose, HR, RR, Age, Gender, BMI, MV duration, APACHE, SEMICYUC, 
Number of previous MV event'''

lstr_cut = lambda string_l: [
    i.split('\n')[1] if '\n' in i else i for i in string_l.split(', ')
]

f_a = np.array(
    lstr_cut(paper_0) + lstr_cut(paper_1) + lstr_cut(paper_2) +
    lstr_cut(paper_3))
unique, counts = np.unique(f_a, return_counts=True)
f_d = dict(zip(unique, counts))


def OccurSelect(dict_, times):

    k_l = []
    for k, v in dict_.items():
        if v >= times:
            k_l.append(k)

    return k_l


print('''
    2 Times Occur: {0}
    3 Times Occur: {1}
    4 Times Occur: {2}
    '''.format(OccurSelect(f_d, 2), OccurSelect(f_d, 3), OccurSelect(f_d, 4)))