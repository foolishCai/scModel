## 傻瓜式评分卡
cr.Cai, 2020-09-07

环境准备
1. python3.6以上
2. scorecardpy，https://github.com/ShichenXie/scorecardpy
3. PrettyTable，https://pypi.org/project/PrettyTable/
-----------------------------------------------------------------
demo简介：
1. 某爸爸提供的假样本数据，风控贷前
2. 特征：固定特征

```python
import warnings
warnings.filterwarnings("ignore")
```

```python
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False
import json

from sklearn.model_selection import train_test_split
```


```python
pro_path = "/Users/cai/Desktop/pythonProjects/github_FoolishCai/scModel"
import sys
import os
sys.path.append(pro_path)

from scToolkits.edaToolkit import EdaUtil as EDA
from scToolkits.FastScToolkit import FastScWoe as FSW
from scToolkits.drawToolkit import plot_learning_curve, get_correlation, get_feature_importance, get_lift_curve, get_plt_kde, get_pr_curve, get_score_bar
```

### step1: 读取数据


```python
df = pd.read_csv("/Users/cai/Desktop/工作projects/1.2.工作备份/测试数据.csv", sep="|", dtype=str)
df = df.drop(['age', 'sex'], axis=1)

df_bak = df.copy()

config_df = pd.read_csv(pro_path + os.sep + "Configs/import_miss_763.txt", sep=",")
config_df = config_df[config_df.in763==1]
del_col = ["number", "id", "id_type", "create_date", "gid", "y"]
serial_col = list(config_df[config_df.if_continuous==1].feature)
serial_col = [i for i in serial_col if i not in del_col and i in df.columns]
dist_col =  list(config_df[config_df.if_continuous==0].feature)
dist_col = [i for i in dist_col if i not in del_col and i in df.columns]
```

#### note1.1：为保持与线上一致，手动填写na


```python
with open(pro_path + os.sep + "Configs/fill_na.json","r") as f:
    naj = json.load(f)

for col in naj.keys():
    df[col].replace(float(naj[col]), np.nan, inplace=True)
    df[col].replace(str(naj[col]), np.nan, inplace=True)
```

### step2: EDA

#### note2.1 EDA基本要求
* 是否有时间单位，观察单位时间内的y1变化情况，判断是否可以作ott验证
* 样本的确实率情况，横纵两个方向，在特征维度上删除覆盖率低于20%的特征（cr. wangmj)
* 不同label下的特征缺失率分布，考虑特征缺失是否可以作为前置规则(cr.chengp)
* 对于离散变量
    * 查找unique value过多的离散特征
    * 查找除了np.nan以外仅有唯一值的离散特征
* 连续变量（查找没啥变化的连续变量）
    * 某一特征非空的连续数值，五分位数与95分位数相等
    * 变异系数(Coefficient of Variation, cv)过小


```python
eda = EDA(file_name="EDA",  df=df, target_name="y", del_col=del_col, dist_col=dist_col, serial_col=serial_col, time_col="create_date",
                 time_level="m", max_col_null_pct=0.8, max_row_null_pct=0.8,
                 max_null_pct_delta=0.1, max_various_values=20)
```

```python
eda.go_init_explore()
```

```python
df = df_bak.copy()
df["y"] = df.y.astype(float).astype(int)
```


```python
df = df.drop(to_drop_cols, axis=1)

if  "ft_lbs_residence_stability" not in to_drop_cols:
    df["ft_lbs_residence_stability"] = df.ft_lbs_residence_stability.map(lambda x: int(bool(x)) if x==x else np.nan)
if "ft_lbs_workplace_stability" not in to_drop_cols:
    df["ft_lbs_workplace_stability"] = df.ft_lbs_workplace_stability.map(lambda x: int(bool(x)) if x==x else np.nan)

_del = [i for i in del_col if i in df.columns]
_serial = [i for i in serial_col if i not in to_drop_cols]
_dist = [i for i in dist_col if i not in to_drop_cols]
```


```python
X_train, X_test, y_train, y_test = train_test_split(df[_serial + _dist + ["y"]], df[["y"]], test_size=0.33, random_state=222, stratify=df.create_date)

print("Train-data: shape={}, y_ratio={}".format(X_train.shape, round(y_train.y.sum()/len(y_train), 4)))
print("Test-data:  shape={}, y_ratio={}".format(X_test.shape, round(y_test.y.sum()/len(y_test), 4)))
```


```python
fsw = FSW(df_train=X_train, target_name="y", dist_col=_dist, serial_col=_serial, df_test=X_test, 
          df_ott=None, max_corr=0.6)
```

```python
fsw.cut_main()
```

```python
fsw.model_main()
```

### step4: 其他指标

#### note4.1 按需采用
* 特征相关系数图示化
* 特征重要性图示化
* 提升曲线图示化
* p-r曲线
* 模型预测概率值分布图示化
* 模型最终业务决策图示化
* 学习曲线


```python
get_correlation(fsw.df_train_woe[fsw.final_features])
```


```python
get_feature_importance(fsw.final_features, fsw.model.coef_[0])
```

```python
get_lift_curve(y_true = fsw.df_test_woe[fsw.target_name], y_pred = fsw.test_pred)
```


```python
get_plt_kde(y_true = fsw.df_test_woe[fsw.target_name], y_pred = fsw.test_pred)
```

```python
get_score_bar(y_true = fsw.df_test_woe[fsw.target_name], y_pred = fsw.test_pred)
```

