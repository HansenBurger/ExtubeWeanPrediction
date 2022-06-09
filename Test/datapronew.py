# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:18:46 2021

@author: qyl
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn import metrics
from xgboost import plot_importance
import matplotlib.pylab as pyplot

from sklearn.model_selection import KFold

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, auc

patient_data = pd.read_csv('病人基本信息.csv')

patient_data['FirstMVTime'] = pd.to_datetime(patient_data['FirstMVTime'])
patient_data['WeanTime'] = pd.to_datetime(patient_data['WeanTime'])

patient_data.drop(patient_data[np.isnan(patient_data['WeanTime'])].index,
                  inplace=True)
patient_data = patient_data.rename(columns={'WeaningStatus': '拔管失败(1:失败)'})
patient_data.loc[patient_data['拔管失败(1:失败)'] == '撤机失败', '拔管失败(1:失败)'] = 1
patient_data.loc[patient_data['拔管失败(1:失败)'] == '撤机成功', '拔管失败(1:失败)'] = 0

pid_lab = patient_data['PID'].unique().tolist()

record_data = pd.read_csv('病人数据表.csv')
record_data['record_time'] = pd.to_datetime(record_data['record_time'])

data = pd.DataFrame()
X_test_final = pd.DataFrame()
datalength = []
a = 0
for pid in pid_lab:
    a = a + 1
    print(str(len(pid_lab) - a) + 'left')
    this_pid_record_data = record_data[record_data.patient_id ==
                                       pid].sort_values(['record_time'])
    this_pid_patient_data = patient_data[patient_data.PID == pid]
    this_pid_patient_data = this_pid_patient_data.reset_index(drop=True)
    weaningtime = this_pid_patient_data.iloc[0, 14]
    intubationtime = this_pid_patient_data.iloc[0, 4]
    this_pid_record_data = this_pid_record_data[
        this_pid_record_data.record_time < weaningtime]
    this_pid_record_data = this_pid_record_data[
        this_pid_record_data.record_time > intubationtime]
    datalength.append(len(this_pid_record_data))
    if (len(this_pid_record_data) >= 48):
        data = data.append(this_pid_record_data)
    else:
        print('drop' + str(pid))

data.to_csv('data.csv', encoding='utf_8_sig')

data['PEEP(cmH2O)'] = data['PEEP(cmH2O)'].replace({'IV': np.nan})
data['PEEP(cmH2O)'] = data['PEEP(cmH2O)'].replace({'NIV': np.nan})
data['PEEP(cmH2O)'] = data['PEEP(cmH2O)'].astype('float32')

col_float = [
    'HR', 'SpO2', 'DBP', 'SBP', 'MBP', 'P/F比值', '乳酸', '二氧化碳分压', '二氧化碳总量', '体温',
    '全血剩余碱', '全血葡萄糖', '总氧饱和度', '总胆红素', '氧分压', '氯', '渗透压', '碳酸氢根浓度', '离子钙',
    '红细胞压积', '给氧浓度', '肺泡动脉氧分压', '血色素', '还原型血红蛋白溶解度', '酸碱度', '钠', '钾', '高铁血红蛋白',
    '呼吸周期(次)', '呼吸频率(b/min)', 'PEEP(cmH2O)', '吸气时间(s)', '压力上升百分比(%)',
    '吸气压力(cmH2O)', 'PS(cmH2O)', 'Esens(%)', 'Vsens(L/min)', 'Psens(cmH2O)',
    'C_STAT(ml/cmH2O)', 'C_DYN(ml/cmH2O)', 'C_STAT_self(ml/cmH2O)', 'DP',
    'DP_self', 'Ppeak(cmH2O)', 'Ppeep(cmH2O)', 'RR(b/min)', 'V_Ti(mL)',
    'V_Te(mL)', 'Fipeak(L/min)', 'Fepeak(L/min)', 'Ve_TOT(L/min)',
    'WOB_A(J/L)', 'WOB_B(J/L)', 'MP_area(J/min)', 'MP_formula(J/min)', 'IEE',
    '异步率1(%)', 'DT', '异步率2(%)', 'Short Cycle', '异步率3(%)', 'Prolonged Cycle',
    '异步率4(%)', 'Async Index(%)'
]

col_object = ['Age', 'Sex', 'BMI', 'APPACHE', 'SOFA', 'Nutic']
col_basic = ['record_time', 'patient_id']
col_flag = ['拔管失败(1:失败)']

l1_std_col = []
l1_mean_col = []
l2_std_col = []
l2_mean_col = []
l3_std_col = []
l3_mean_col = []

for col in col_float:
    l1_std_col.append('l1_' + col + '_std')
    l1_mean_col.append('l1_' + col + '_mean')
    l2_std_col.append('l2_' + col + '_std')
    l2_mean_col.append('l2_' + col + '_mean')
    l3_std_col.append('l3_' + col + '_std')
    l3_mean_col.append('l3_' + col + '_mean')

final_col = col_basic + l1_std_col + l1_mean_col + l2_std_col + l2_mean_col + l3_std_col + l3_mean_col
data_tmp = data[col_basic + col_float]

pid_lab = data_tmp['patient_id'].unique().tolist()
final_data = []
for pid in pid_lab:
    print(pid)
    this_pid_data = data_tmp[data_tmp.patient_id == pid].sort_values(
        ['record_time'])
    basic_data = this_pid_data[col_basic].drop_duplicates(
    ).iloc[0].values.tolist()

    l1_data = this_pid_data.iloc[0:24, 2:]

    l1_std = np.nanstd(l1_data[col_float], axis=0).tolist()
    l1_mean = np.nanmean(l1_data[col_float], axis=0).tolist()

    l3_data = this_pid_data.iloc[-24:, 2:]

    l3_std = np.nanstd(l1_data[col_float], axis=0).tolist()
    l3_mean = np.nanmean(l1_data[col_float], axis=0).tolist()

    l2_data = this_pid_data.iloc[-36:-12, 2:]
    l2_std = np.nanstd(l2_data[col_float], axis=0).tolist()
    l2_mean = np.nanmean(l2_data[col_float], axis=0).tolist()

    temp = basic_data + l1_std + l1_mean + l2_std + l2_mean + l3_std + l3_mean

    final_data.append(temp)

final_data = pd.DataFrame(final_data, columns=final_col)
final_data = final_data.replace({'nan': np.nan})
final_data = final_data.rename(columns={'patient_id': 'PID'})
merge_data = pd.merge(patient_data, final_data, how='right', on=['PID'])
merge_data = merge_data.rename(columns={'PID': 'patient_id'})

X_data = merge_data[col_object + col_flag + final_col]
X_data = X_data.drop(columns=['record_time', 'patient_id'])

X_data.loc[X_data['Sex'] == '女', 'Sex'] = 0
X_data.loc[X_data['Sex'] == '男', 'Sex'] = 1
zscore_scale_num = preprocessing.StandardScaler()
stabdard_col = l1_std_col + l1_mean_col + l2_std_col + l2_mean_col + l3_std_col + l3_mean_col + col_object
X_data[stabdard_col] = zscore_scale_num.fit_transform(X_data[stabdard_col])

len_old = X_data.shape[1]
X_data = X_data.dropna(axis=1, how='all')
print('drop ' + str(len_old - X_data.shape[1]) + ' all nan features')

threshold = len(X_data) * 0.4
len_old = X_data.shape[1]
X_data = X_data.dropna(axis=1, thresh=threshold)
print('drop ' + str(len_old - X_data.shape[1]) + ' >40% nan features')
print('now: ' + str(X_data.shape[1] - 2) + ' features')

#删除横向缺失>80%样本
threshold = len(X_data.columns) * 0.8
len_old1 = X_data[X_data['拔管失败(1:失败)'] == 1].shape[0]
len_old0 = X_data[X_data['拔管失败(1:失败)'] == 0].shape[0]
X_data1 = X_data.dropna(axis=0, thresh=threshold)
X_data = X_data.dropna(axis=0, thresh=threshold)
len_new1 = X_data1[X_data1['拔管失败(1:失败)'] == 1].shape[0]
len_new0 = X_data1[X_data1['拔管失败(1:失败)'] == 0].shape[0]
print('drop ' + str(len_old1 - len_new1) + ' >80% nan positive samples')
print('now: ' + str(len_new1) + ' positive samples')
print('drop ' + str(len_old0 - len_new0) + ' >80% nan negtive samples')
print('now: ' + str(len_new0) + ' negtive samples')

X_data = X_data.reset_index(drop=True)
df_0 = X_data[X_data['拔管失败(1:失败)'] == 0]
df_1 = X_data[X_data['拔管失败(1:失败)'] == 1]

KF = KFold(n_splits=5, shuffle=True, random_state=104)

train_index_0 = []
val_index_0 = []
train_index_1 = []
val_index_1 = []

f1 = open('result1/all.txt', 'w')
for train_index0, val_index0 in KF.split(df_0):  #阴性样本训练验证集索引划分
    train_index_0.append(train_index0)
    val_index_0.append(val_index0)
for train_index1, val_index1 in KF.split(df_1):  #阳性样本训练验证集索引划分
    train_index_1.append(train_index1)
    val_index_1.append(val_index1)

for j in range(0, 5):
    train_index0 = train_index_0[j]  #提取第j折阴性样本训练集索引
    train_index1 = train_index_1[j]  #提取第j折阳性样本训练集索引
    val_index0 = val_index_0[j]  #提取第j折阴性样本验证集索引
    val_index1 = val_index_1[j]  #提取第j折阳性样本验证集索引
    k = j + 1
    X_Y_train_0 = df_0.iloc[train_index0, :]  #提取阴性样本训练数据
    X_Y_train_1 = df_1.iloc[train_index1, :]  #提取阳性样本训练数据
    #重采样平衡阳性阴性样本数量

    X_Y_train_0 = resample(
        X_Y_train_0,
        replace=True,  # sample with replacement
        n_samples=len(X_Y_train_1) * 2,  # to match majority class
        random_state=300)  # reproducible results

    X_Y_train_1 = resample(
        X_Y_train_1,
        replace=True,  # sample with replacement
        n_samples=len(X_Y_train_0),  # to match majority class
        random_state=300)  # reproducible results

    # 合并阴性阳性训练样本

    X_Y_train = pd.concat([X_Y_train_0, X_Y_train_1])
    X_Y_train = X_Y_train.sample(frac=1)
    X_Y_train = X_Y_train.reset_index(drop=True)
    X_train_kfold = X_Y_train.copy().drop('拔管失败(1:失败)', axis=1)  #训练集删去标签
    Y_train_kfold = X_Y_train[['拔管失败(1:失败)']]  #提取训练集标签

    X_Y_val_0 = df_0.iloc[val_index0, :]
    X_Y_val_1 = df_1.iloc[val_index1, :]
    X_Y_val = pd.concat([X_Y_val_0, X_Y_val_1])
    X_Y_val = X_Y_val.sample(frac=1)
    X_Y_val = X_Y_val.reset_index(drop=True)
    X_val_kfold = X_Y_val.copy().drop('拔管失败(1:失败)', axis=1)
    Y_val_kfold = X_Y_val[['拔管失败(1:失败)']]

    X_train_kfold = X_train_kfold.astype('float32')
    X_val_kfold = X_val_kfold.astype('float32')
    Y_train_kfold = Y_train_kfold.astype('float32')
    Y_val_kfold = Y_val_kfold.astype('float32')
    data_train = xgb.DMatrix(X_train_kfold, Y_train_kfold)
    data_test = xgb.DMatrix(X_val_kfold, Y_val_kfold)
    eval_set = [(X_train_kfold, Y_train_kfold), (X_val_kfold, Y_val_kfold)]
    params = {
        'booster': ["gbtree"],
        'learning_rate': [
            0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
            0.45, 0.5
        ],
        'n_estimators':
        range(100, 6000, 100),
        'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
        'gamma': [0, 1e-1, 1, 5, 10, 20, 50, 100],
        'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'max_depth': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "scale_pos_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'reg_alpha': [1, 0.5, 0.1, 0.08]
    }
    xlf = XGBClassifier(
        silent=False,  # =0打印信息
        learning_rate=0.2,  # 默认
        subsample=0.8,  # 随机采样训练样本 训练实例的子采样比
        colsample_bytree=0.8,  # 生成树时进行的列采样
        objective='binary:logistic',  # 指定学习任务和相应的学习目标
        n_estimators=200,  # 树的个数
        max_depth=5,  # 构建树的深度，越大越容易过拟合
        min_child_weight=
        1,  # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        reg_alpha=0.08,
        # reg_lambda=0.01,
        nthread=4,  # cpu线程数
        scale_pos_weight=1,  # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        seed=27,  # 随机种子
        eval_metric=['logloss', 'auc', 'error'])

    gsearch = RandomizedSearchCV(xlf,
                                 param_distributions=params,
                                 scoring='accuracy',
                                 cv=3)

    gsearch.fit(X_train_kfold,
                Y_train_kfold,
                early_stopping_rounds=10,
                eval_set=eval_set,
                verbose=10)

    params = gsearch.best_params_
    clf = XGBClassifier(
        silent=False,  # =0打印信息
        learning_rate=params['learning_rate'],  # 默认
        subsample=params['subsample'],  # 随机采样训练样本 训练实例的子采样比
        colsample_bytree=params['colsample_bytree'],  # 生成树时进行的列采样
        objective='binary:logistic',  # 指定学习任务和相应的学习目标
        n_estimators=params['n_estimators'],  # 树的个数
        max_depth=params['max_depth'],  # 构建树的深度，越大越容易过拟合
        min_child_weight=params[
            'min_child_weight'],  # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        gamma=params['gamma'],  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        reg_alpha=params['reg_alpha'],
        # reg_lambda=0.01,
        nthread=4,  # cpu线程数
        scale_pos_weight=params[
            'scale_pos_weight'],  # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        seed=27,  # 随机种子
        eval_metric=['logloss', 'auc', 'error'])

    clf.fit(X_train_kfold,
            Y_train_kfold,
            early_stopping_rounds=10,
            eval_set=eval_set,
            verbose=10)
    #   删选important_features
    feature_names = X_train_kfold.columns
    feature_importance = pd.DataFrame(
        pd.Series(clf.get_booster().get_fscore()),
        columns=['importance_of_' + str(k) + '_fold'])

    important_features = feature_importance.loc[
        feature_importance['importance_of_' + str(k) + '_fold'] > 0, :].index
    X_train_final = X_train_kfold[important_features]
    X_val_final = X_val_kfold[important_features]
    X_train_final = X_train_final.astype('float32')
    X_val_final = X_val_final.astype('float32')
    data_train = xgb.DMatrix(X_train_final, Y_train_kfold)
    data_test = xgb.DMatrix(X_val_final, Y_val_kfold)
    eval_set = [(X_train_final, Y_train_kfold), (X_val_final, Y_val_kfold)]

    clf.fit(X_train_final,
            Y_train_kfold,
            early_stopping_rounds=50,
            eval_set=eval_set,
            verbose=10)

    y_pre = clf.predict(X_val_final)
    y_pro = clf.predict_proba(X_val_final)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(Y_val_kfold, y_pro)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积

    plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

    #开始画ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('ROC')
    plt.savefig('./result1/ROC' + str(k) + '.png', dpi=300)
    plt.close()

    thres = 0.5
    TP = 0  #Correctly detected as arrhythmia
    TN = 0  #Correctly detected as normal
    FP = 0  #Incorrectly detected as arrhythmia
    FN = 0  #Incorrectly detected as normal
    for num in range(len(Y_val_kfold)):
        if y_pro[num] >= thres:
            label = 1
        else:
            label = 0
        if Y_val_kfold.values[num] == 1:
            if label == 1:
                TP = TP + 1
            # save_path = './predict/TP/'
            # shutil.copy(X_test[num], save_path)
            else:
                FN = FN + 1
            # save_path = './predict/FN/'
            # shutil.copy(X_test[num], save_path)

        if Y_val_kfold.values[num] == 0:
            if label == 0:
                TN = TN + 1
            # save_path = './predict/TN/'
            # shutil.copy(X_test[num], save_path)
            else:
                FP = FP + 1
            # save_path = './predict/FP/'
            # shutil.copy(X_test[num], save_path)

    Accuray = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Specificity = TN / (FP + TN)
    Sensitivity = TP / (TP + FN)
    PositivePredictiveValue = TP / (TP + FP)
    F1_Score = 2 * Precision * Recall / (Precision + Recall)
    f1.write('第' + str(k) + '折\n')
    f1.write('Accuray: ' + str(Accuray * 100) + '%' + '\n')
    f1.write('Precision: ' + str(Precision * 100) + '%' + '\n')
    f1.write('Recall: ' + str(Recall * 100) + '%' + '\n')
    # f1.write('Specificity: '+str(Specificity * 100)+'\n')
    # f1.write('Sensitivity: '+str(Sensitivity * 100)+'\n')
    # f1.write('PositivePredictiveValue: '+str(PositivePredictiveValue * 100)+'\n')
    f1.write('F1_Score: ' + str(F1_Score) + '\n')
    f1.write("AUC Score : " + str(metrics.roc_auc_score(Y_val_kfold, y_pro)) +
             '\n')
    f1.write('==========================================\n')
    f1.flush()

    results = clf.evals_result()

    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')

    plt.savefig('./result2/Log Loss' + str(k) + '.png', dpi=300)
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')

    plt.savefig('./result2/Classification Error' + str(k) + '.png', dpi=300)
    plt.close()

    from xgboost import plot_importance
    import matplotlib.pyplot as plt

    feature_importance['importance'] = feature_importance.mean(axis=1)

    feature_importances = pd.DataFrame({
        'feature':
        feature_importance.index,
        'importance':
        feature_importance.importance,
    }).sort_values(by=['importance'], ascending=True)

    feature_importances.loc[:, 'color'] = 'lightgrey'

    l1 = 0
    l2 = 0
    l3 = 0

    for kk in range(0, len(feature_importances)):
        features = feature_importances.iloc[kk, :].feature
        #标记d1特征
        if (features.startswith('l1')):
            feature_importances.loc[features, 'color'] = 'lightpink'
            l1 = l1 + feature_importances.loc[features, 'importance']
    #标记l1特征
        if (features.startswith('l2')):
            feature_importances.loc[features, 'color'] = 'lightgreen'
            l2 = l2 + feature_importances.loc[features, 'importance']
        if (features.startswith('l3')):
            feature_importances.loc[features, 'color'] = 'lightblue'
            l3 = l3 + feature_importances.loc[features, 'importance']

#    else:
#        feature_importances.loc[features,'color']='lightgrey'
#
    plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号
    plt.rcParams['font.family'] = 'SimHei'  #用来正常显示负号
    plt.style.use('seaborn')
    plt.figure(figsize=(20, 30))
    num = 50
    plt.barh(range(len(feature_importances[-num:])),
             feature_importances[-num:].importance,
             tick_label=feature_importances[-num:].feature,
             color=feature_importances[-num:].color)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    #plt.title('Feature Importance Of Xgboost',fontsize=30)  # 图形标题
    ax = plt.gca()
    #获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.savefig('./result1/fi' + str(k) + '.png', dpi=300)
    plt.close()
f1.close()

f1.close()
