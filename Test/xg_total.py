from pickle import FALSE
import sys
import xgboost as xgb
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import resample
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score, auc
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(str(Path.cwd()))

from Classes.balancefunc import SMOTE_
from Classes.Func.KitTools import SaveGen
from Classes.ORM.expr import PatientInfo, LabExtube, LabWean
from Classes.ORM.cate import ExtubePSV, ExtubeSumP12, WeanPSV, WeanSumP12

mode_ = 'Extube_SumP12_Nad_vt_mp_xgboost'  # run nad mode then
data_p = r'C:\Main\Data\_\Result\Form\20220509_19_Extube_SumP12_Nad'

mode_info = {
    'Extube': {
        'Lab': LabExtube,
        'PSV': ExtubePSV,
        'SumP12': ExtubeSumP12
    },
    'Wean': {
        'Lab': LabWean,
        'PSV': WeanPSV,
        'SumP12': WeanSumP12
    }
}

save_loc = SaveGen(r'C:\Main\Data\_\Result\Graph', mode_)


def main():
    # df_0, df_1 = DataLoader(data_p)
    df = pd.read_csv(
        r'C:\Users\HY_Burger\Desktop\Project\ExtubeWeanPrediction\basedata.csv'
    )
    df_0 = df[df.end == 0]
    df_1 = df[df.end == 1]
    MainPorcess(df_0, df_1)
    pass


def DataLoader(load_path: str):
    p_i_l = []
    m_i_l = mode_.split('_')

    for path in Path(load_path).iterdir():
        if not path.is_file():
            pass
        else:
            p_r = pd.read_csv(path, index_col='method')
            p_info = path.name.split('_')
            p_r_ave = p_r.loc['ave'].to_dict()
            p_i_d = {'pid': p_info[0], 'end': int(p_info[1]), 'rid': p_info[2]}
            p_i_d.update(p_r_ave)
            p_i_l.append(p_i_d)

    df_basic = pd.DataFrame(p_i_l)

    src_0, src_1 = PatientInfo, mode_info[m_i_l[0]]['Lab']
    join_info = {'dest': src_0, 'on': src_0.pid == src_1.pid, 'attr': 'pinfo'}
    col_que = [src_1, src_0.age, src_0.sex, src_0.bmi]
    col_order = [src_1.pid]
    cond = src_1.pid.in_(df_basic.pid.to_list())
    que_l = src_1.select(*col_que).join(**join_info).where(cond).order_by(
        *col_order)

    df_que = pd.DataFrame(list(que_l.dicts()))
    df_que = df_que.drop('pid', axis=1)

    df_total = pd.concat([df_basic, df_que], axis=1)
    df_total = df_total.drop(['pid', 'rid'], axis=1)

    # drop featutre nan > 40%
    df_total = DropByThreshold(df_total, 0.4, 1)
    # drop data nan > 80%
    df_total = DropByThreshold(df_total, 0.8, 0)

    df_0 = df_total[df_total.end == 0]
    df_1 = df_total[df_total.end == 1]

    pd.DataFrame.to_csv(df_total, 'Feature_base.csv', index=False)

    return df_0, df_1


def DropByThreshold(df: pd.DataFrame, per: float, ax_st: int):
    ax_set = int(not ax_st)
    threshold = df.shape[ax_set] * per
    len_raw = df.shape[ax_st]
    df = df.dropna(axis=ax_st, thresh=threshold)
    len_new = df.shape[ax_st]
    print('Drop col/row: {0}'.format(len_raw - len_new))
    return df


def MainPorcess(df_0, df_1):

    train_index_0 = []
    val_index_0 = []
    train_index_1 = []
    val_index_1 = []
    result_1 = save_loc / 'result1'
    result_1.mkdir(parents=True, exist_ok=True)
    f1 = open((result_1 / 'all.txt'), 'w')

    KF = KFold(n_splits=5, shuffle=True, random_state=104)

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
        # X_Y_train_0 = resample(
        #     X_Y_train_0,
        #     replace=True,  # sample with replacement
        #     n_samples=len(X_Y_train_1) * 3,  # to match majority class
        #     random_state=300)  # reproducible results

        # X_Y_train_1 = resample(
        #     X_Y_train_1,
        #     replace=True,  # sample with replacement
        #     n_samples=len(X_Y_train_0),  # to match majority class
        #     random_state=300)  # reproducible results
        # 合并阴性阳性训练样本

        X_Y_train = pd.concat([X_Y_train_0, X_Y_train_1])
        X_Y_train = X_Y_train.sample(frac=1)
        X_Y_train = X_Y_train.reset_index(drop=True)
        X_train_kfold = X_Y_train.copy().drop('end', axis=1)  #训练集删去标签
        Y_train_kfold = X_Y_train[['end']]  #提取训练集标签

        p_smote = SMOTE_(X_train_kfold, Y_train_kfold)
        p_smote.OverSample()
        X_train_kfold = p_smote.X
        Y_train_kfold = p_smote.y

        X_Y_val_0 = df_0.iloc[val_index0, :]
        X_Y_val_1 = df_1.iloc[val_index1, :]
        X_Y_val = pd.concat([X_Y_val_0, X_Y_val_1])
        X_Y_val = X_Y_val.sample(frac=1)
        X_Y_val = X_Y_val.reset_index(drop=True)
        X_val_kfold = X_Y_val.copy().drop('end', axis=1)
        Y_val_kfold = X_Y_val[['end']]

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
            'min_child_weight':
            [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
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
            gamma=params[
                'gamma'],  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
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
            feature_importance['importance_of_' + str(k) +
                               '_fold'] > 0, :].index
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
        save_n_roc = result_1 / ('ROC' + str(k) + '.png')
        plt.savefig(save_n_roc, dpi=300)
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
        if (Precision + Recall) == 0:
            F1_Score = 0
        else:
            F1_Score = 2 * Precision * Recall / (Precision + Recall)
        repr_gen = lambda dict_: ('\n\t').join(k + ':\t' + str(v)
                                               for k, v in dict_.items())
        f1.write('第' + str(k) + '折\n')
        f1.write('Accuray: ' + str(Accuray * 100) + '%' + '\n')
        f1.write('Precision: ' + str(Precision * 100) + '%' + '\n')
        f1.write('Recall: ' + str(Recall * 100) + '%' + '\n')
        # f1.write('Specificity: '+str(Specificity * 100)+'\n')
        # f1.write('Sensitivity: '+str(Sensitivity * 100)+'\n')
        # f1.write('PositivePredictiveValue: '+str(PositivePredictiveValue * 100)+'\n')
        f1.write('F1_Score: ' + str(F1_Score) + '\n')
        f1.write("AUC Score : " +
                 str(metrics.roc_auc_score(Y_val_kfold, y_pro)) + '\n')
        f1.write('Params: \n' + repr_gen(params))
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

        result_2 = save_loc / 'result2'
        result_2.mkdir(parents=True, exist_ok=True)
        save_n_loss = result_2 / ('Log Loss' + str(k) + '.png')

        plt.savefig(save_n_loss, dpi=300)
        # plot classification error
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        plt.ylabel('Classification Error')
        plt.title('XGBoost Classification Error')

        save_n_error = result_2 / ('Classification Error' + str(k) + '.png')

        plt.savefig(save_n_error, dpi=300)
        plt.close()

        feature_importance['importance'] = feature_importance.mean(axis=1)

        feature_importances = pd.DataFrame({
            'feature':
            feature_importance.index,
            'importance':
            feature_importance.importance,
        }).sort_values(by=['importance'], ascending=True)

        feature_importances.loc[:, 'color'] = 'lightgrey'

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
        save_n_fi = result_1 / ('fi' + str(k) + '.png')
        plt.savefig(save_n_fi, dpi=300)
        plt.close()


if __name__ == '__main__':
    main()