#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 09:52
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : drawToolkit.py
# @Note    :


import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd
from sklearn.model_selection import learning_curve


def get_badpct_by_time(df, file_name=None):
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(111)

    plt1 = ax1.bar(df.index, df['total'], color="yellow", width=0.4, label="total")

    ax2 = ax1.twinx()  # this is the important function
    plt2, = ax2.plot(df.index, df['bad_pct'], color='green', linewidth=5, label="bad_pct")

    for tl in ax1.get_xticklabels():
        tl.set_rotation(45)
        tl.set_fontsize(8)

    ax1.set(xlabel='时间索引', ylabel="total")
    ax2.set(ylabel="bad_pct")
    plt.title('样本随时间分布')
    plt.legend([plt1, plt2], ('total', 'bad_pct',), loc='upper right')
    plt.show()
    if file_name:
        plt.savefig(file_name + ".png")


def get_null_pct_by_label(df, file_name=None):
    plt.figure(figsize=(20, 7))
    plt.plot(df['label_0'], 'r', label='y=0')
    plt.plot(df['label_1'], 'g', label='y=1')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()
    if file_name:
        plt.savefig(file_name+".png", dpi=500)

def get_all_null(df, file_name=None):
    plt.figure(figsize=(15, 5))
    plt.scatter(df.num, df.null_pct, c='b')
    plt.xlim([0, len(df.index)])
    plt.xlabel('样本排序')
    plt.ylabel('null值占比')
    plt.title('distribution of null nums')
    plt.show()
    if file_name:
        plt.savefig(file_name + '.png', dpi=500)

def get_correlation(df, file_name=None):
    """
    :param df: pandas.Dataframe
    :param figsize:
    :return:
    """

    plt.figure(figsize=(20, 20))
    colormap = plt.cm.viridis
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.astype(float).corr(),
                linewidths=0.1,
                vmax=1.0,
                square=True,
                cmap=colormap,
                linecolor='white',
                annot=True)
    plt.show()
    if file_name:
        plt.savefig(file_name + '.png', dpi=500)


def get_curve(y_true, y_pred, file_name):

        plt.figure(figsize=(12, 6))
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.subplot(121)
        plt.xlabel('Percentage', fontsize=15)
        plt.ylabel('tpr / fpr', fontsize=15)
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.title("ks-curve", fontsize=20)

        percentage = np.round(np.array(range(1, len(fpr) + 1)) / len(fpr), 4)
        ks_delta = tpr - fpr
        ks_index = ks_delta.argmax()
        plt.plot([percentage[ks_index], percentage[ks_index]],
                 [tpr[ks_index], fpr[ks_index]],
                 color='limegreen', lw=2, linestyle='--')
        plt.text(percentage[ks_index] + 0.02, (tpr[ks_index] + fpr[ks_index]) / 2,
                 'ks: {0:.4f}'.format(ks_delta[ks_index]),
                 fontsize=13)
        plt.plot(percentage, tpr, color='dodgerblue', lw=2, label='tpr')
        plt.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
        plt.plot(percentage, fpr, color='tomato', lw=2, label='fpr')
        plt.legend(fontsize='x-large')

        plt.subplot(122)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.savefig(file_name)
        plt.show()


def get_feature_importance(feature, importance, top=30, filename=None):
    if len(feature) < top:
        top = len(feature)
    d = dict(zip(feature, importance))
    feature_importance_list = sorted(d.items(), key=lambda item: abs(item[1]), reverse=True)
    top_names = [i[0] for i in feature_importance_list][: top]

    plt.figure(figsize=(8, 6))
    plt.title("Feature importances")
    plt.barh(range(top), [d[i] for i in top_names], color="b", align="center")
    plt.ylim(-1, top)
    plt.xlim(min(importance), max(importance))
    plt.yticks(range(top), top_names)
    plt.show()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', format='png', dpi=300, pad_inches=0, transparent=True)
    # return feature_importance_list[:top]


def get_cm(y_true, y_pred, thresh, file_name=None):
    cm = confusion_matrix(list(y_true), [int(i>thresh) for i in list(y_pred)])   # 由原标签和预测标签生成混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.title('confusion matrix')
    plt.show()
    if file_name is not None:
        plt.savefig(file_name)



def plot_learning_curve(estimator, X, y, file_name=None):
    plt.figure()
    plt.title("learning_curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    if file_name:
        plt.savefig(file_name + '.png', dpi=500)



def get_correlation(df, file_name=None):
    plt.figure(figsize=(20, 20))
    colormap = plt.cm.viridis
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.astype(float).corr(),
                linewidths=0.1,
                vmax=1.0,
                square=True,
                cmap=colormap,
                linecolor='white',
                annot=True)
    plt.show()
    if file_name:
        plt.savefig(file_name + '.png', dpi=500)


def get_lift_curve(y_true, y_pred, file_name=None):
    result = pd.DataFrame(columns=['target', 'proba'], data={"target": list(y_true), "proba": list(y_pred)})
    result_ = result.copy()
    proba_copy = result.proba.copy()
    for i in range(10):
        point1 = scoreatpercentile(result_.proba, i * (100 / 10))
        point2 = scoreatpercentile(result_.proba, (i + 1) * (100 / 10))
        proba_copy[(result_.proba >= point1) & (result_.proba <= point2)] = ((i + 1))
    result_['grade'] = proba_copy
    df_gain = result_.groupby(by=['grade'], sort=True).sum() / (len(result) / 10) * 100
    plt.plot(df_gain['target'], color='red')
    for xy in zip(df_gain['target'].reset_index().values):
        plt.annotate("%s" % round(xy[0][1], 2), xy=xy[0], xytext=(-20, 10), textcoords='offset points')
    plt.plot(list(df_gain.index), [round(result.target.sum() / len(result), 4) * 100] * len(list(df_gain.index)),
             color="blue")
    plt.title('Lift Curve')
    plt.xlabel('Decile')
    plt.ylabel('Lift Value')
    plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    if file_name:
        plt.savefig(file_name + '.png', dpi=500)


def get_plt_kde(y_true, y_pred, file_name=None):
    result = pd.DataFrame(columns = ['target','proba'], data={"target":list(y_true), "proba":list(y_pred)})
    plt.figure(figsize=(8, 5))
    sns.kdeplot(result[result.target == 0].proba, label = 'label == 0',lw=2, shade=True)
    sns.kdeplot(result[result.target == 1].proba, label = 'label == 1',lw=2, shade=True)
    plt.title('Kernel Density Estimation')
    plt.show()
    if file_name:
        plt.savefig(file_name + '.png', dpi=500)



def get_pr_curve(y_true, y_pred, file_name=None):
    precision, recall, thr = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('P-R curve', fontdict=font)
    plt.fill_between(recall, precision, alpha=0.5, color='0.75')
    plt.grid(True, linestyle='-', color='0.75')
    plt.plot(recall, precision, lw=1, color='#FF0000')
    if file_name:
        plt.savefig(file_name + ".png")

# 区间正负占比
def get_score_bar(y_true, y_pred, file_name=None):
    plt.figure(figsize=(12, 6))
    rs = pd.DataFrame(columns=['flag', 'pred'], data={"flag": list(y_true), "pred": [round(i,3) for i in list(y_pred)]})
    rs["pred_bin"] = pd.cut(rs["pred"].tolist(), 20, include_lowest=True)
    rs = pd.crosstab(rs.pred_bin, rs.flag, rownames=['pred_bin'], colnames=['flag']).reset_index()
    rs.columns = ["pred_bin", "good", "bad"]
    plt.bar(x=list(rs.pred_bin.astype(str)), height=rs.good, label='y0', alpha=0.8)
    plt.bar(x=list(rs.pred_bin.astype(str)), height=rs.bad, label='y1', alpha=0.8, bottom=rs.good)
    plt.xticks(rotation=90)
    plt.legend(["good", "bad"])
    plt.ylabel("num")
    plt.show()
    if file_name:
        plt.savefig(file_name + "_num.png")

    plt.figure(figsize=(12, 6))
    rs["good_ratio"] = rs["good"] / (rs["good"] + rs["bad"])
    rs["bad_ratio"] = rs["bad"] / (rs["good"] + rs["bad"])
    plt.bar(x=list(rs.pred_bin.astype(str)), height=rs.good_ratio, label='y0', alpha=0.8)
    plt.bar(x=list(rs.pred_bin.astype(str)), height=rs.bad_ratio, label='y1', alpha=0.8, bottom=rs.good_ratio)
    plt.xticks(rotation=90)
    plt.legend(["good", "bad"])
    plt.ylabel("ratio")
    plt.show()
    if file_name:
        plt.savefig(file_name + "_ratio.png")


def plot_learning_curve(estimator, X, y):
    plt.figure()
    plt.title("learning_curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()