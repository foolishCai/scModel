#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 09:52
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : drawToolkit.py
# @Note    :


import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np



from sklearn.model_selection import learning_curve
def get_badpct_by_time(df, file_name=None):
    plt.figure(figsize=(8, 5))
    df['total'].plot(kind='bar', color="yellow", width=0.4)
    plt.ylabel('总数')
    plt.title('样本随时间分布')
    plt.legend(["总数"])
    df['bad_pct'].plot(color='green', secondary_y=True, style='-o', linewidth=5)
    plt.ylabel('坏样本（比例）')
    plt.legend(["比例"])
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
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', format='png', dpi=300, pad_inches=0, transparent=True)
    plt.show()
    return feature_importance_list[:top]


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
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    return plt

# plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)