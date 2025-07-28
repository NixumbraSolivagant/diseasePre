# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import shutil

# 设置 matplotlib 使用绝对路径的中文字体
FONT_PATH = './AaXinRui85-2.ttf'
font_prop = FontProperties(fname=FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_DIR = './附件'
CSV_OUT_DIR = './output/csv/第一问/'
PLT_OUT_DIR = './output/plt/第一问/'

# 自动创建输出目录
os.makedirs(CSV_OUT_DIR, exist_ok=True)
os.makedirs(PLT_OUT_DIR, exist_ok=True)

# 1. 读取数据
BACKUP_DIR = os.path.join(DATA_DIR, 'backup')
os.makedirs(BACKUP_DIR, exist_ok=True)

cirr_path = os.path.join(DATA_DIR, 'cirrhosis.csv')
heart_path = os.path.join(DATA_DIR, 'heart.csv')
stroke_path = os.path.join(DATA_DIR, 'stroke.csv')

# 备份原始数据
shutil.copy2(cirr_path, os.path.join(BACKUP_DIR, 'cirrhosis.csv'))
shutil.copy2(heart_path, os.path.join(BACKUP_DIR, 'heart.csv'))
shutil.copy2(stroke_path, os.path.join(BACKUP_DIR, 'stroke.csv'))

# 读取数据
df_cirr = pd.read_csv(cirr_path, encoding='utf-8-sig')
df_heart = pd.read_csv(heart_path, encoding='utf-8-sig')
df_stroke = pd.read_csv(stroke_path, encoding='utf-8-sig')

# ========== cirrhosis.csv 预处理 ==========
# 年龄单位转换（天转年）
df_cirr['Age'] = (df_cirr['Age'] // 365).astype('Int64')

# 统一缺失值
for df in [df_cirr, df_heart, df_stroke]:
    df.replace(['NA', 'na', '', ' '], np.nan, inplace=True)

# 用众数填充缺失值
def fill_mode(df):
    for col in df.columns:
        mode = df[col].mode()
        if not mode.empty:
            df[col].fillna(mode[0], inplace=True)
    return df

df_cirr = fill_mode(df_cirr)
df_heart = fill_mode(df_heart)
df_stroke = fill_mode(df_stroke)

# cirrhosis 异常值处理（年龄>110）
df_cirr = df_cirr[df_cirr['Age'] <= 110]

# 保存清洗后的数据到 ./附录
clean_cirr_path = os.path.join(DATA_DIR, 'cirrhosis.csv')
clean_heart_path = os.path.join(DATA_DIR, 'heart.csv')
clean_stroke_path = os.path.join(DATA_DIR, 'stroke.csv')
df_cirr.to_csv(clean_cirr_path, index=False, encoding='utf-8-sig')
df_heart.to_csv(clean_heart_path, index=False, encoding='utf-8-sig')
df_stroke.to_csv(clean_stroke_path, index=False, encoding='utf-8-sig')

# ========== heart.csv 异常值处理 ==========
# 以医学常识为例：年龄>110、收缩压<50或>250、胆固醇<50或>600
if 'Age' in df_heart.columns:
    df_heart = df_heart[df_heart['Age'].astype(float) <= 110]
if 'RestingBP' in df_heart.columns:
    df_heart = df_heart[(df_heart['RestingBP'].astype(float) >= 50) & (df_heart['RestingBP'].astype(float) <= 250)]
if 'Cholesterol' in df_heart.columns:
    df_heart = df_heart[(df_heart['Cholesterol'].astype(float) >= 50) & (df_heart['Cholesterol'].astype(float) <= 600)]

# ========== stroke.csv 异常值处理 ==========
# 年龄>110，BMI>60
if 'age' in df_stroke.columns:
    df_stroke = df_stroke[df_stroke['age'].astype(float) <= 110]
if 'bmi' in df_stroke.columns:
    df_stroke = df_stroke[df_stroke['bmi'].replace('N/A', np.nan).astype(float) <= 60]

# ========== 特征工程 ==========
def encode_and_scale(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    # 分类变量编码
    cat_cols = df.select_dtypes(include=['object']).columns.difference(exclude_cols)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # 数值变量标准化
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    return df

# cirrhosis.csv
cirr_exclude = ['Stage'] if 'Stage' in df_cirr.columns else []
df_cirr_proc = encode_and_scale(df_cirr, exclude_cols=cirr_exclude)
# heart.csv
heart_exclude = ['HeartDisease'] if 'HeartDisease' in df_heart.columns else []
df_heart_proc = encode_and_scale(df_heart, exclude_cols=heart_exclude)
# stroke.csv
stroke_exclude = ['stroke'] if 'stroke' in df_stroke.columns else []
df_stroke_proc = encode_and_scale(df_stroke, exclude_cols=stroke_exclude)

# ========== 描述性统计 ==========
def describe_and_save(df, name):
    desc = df.describe(include='all')
    desc.to_csv(os.path.join(CSV_OUT_DIR, f'{name}_describe.csv'), encoding='utf-8-sig')
    print(f'{name} 描述性统计已保存')

describe_and_save(df_cirr, 'cirrhosis')
describe_and_save(df_heart, 'heart')
describe_and_save(df_stroke, 'stroke')

# ========== 相关性热力图 ==========
def plot_corr(df, name):
    plt.figure(figsize=(12, 8))
    corr = df.corr(method='spearman')
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f', square=True, cbar=True)
    plt.title(f'{name} 相关性热力图', fontproperties=font_prop)
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_OUT_DIR, f'{name}_corr.png'))
    plt.close()

plot_corr(df_cirr_proc, 'cirrhosis')
plot_corr(df_heart_proc, 'heart')
plot_corr(df_stroke_proc, 'stroke')

# ========== 患病率条形图 ==========
def plot_rate_bar(df, label_col, name):
    plt.figure(figsize=(6,4))
    ax = sns.countplot(x=label_col, data=df)
    ax.set_title(f'{name} 患病率分布', fontproperties=font_prop)
    ax.set_xlabel(label_col, fontproperties=font_prop)
    ax.set_ylabel('计数', fontproperties=font_prop)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop)
    if ax.get_legend() is not None:
        ax.legend(prop=font_prop)
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_OUT_DIR, f'{name}_rate.png'))
    plt.close()

if 'Stage' in df_cirr.columns:
    plot_rate_bar(df_cirr, 'Stage', 'cirrhosis')
if 'HeartDisease' in df_heart.columns:
    plot_rate_bar(df_heart, 'HeartDisease', 'heart')
if 'stroke' in df_stroke.columns:
    plot_rate_bar(df_stroke, 'stroke', 'stroke')

# ========== 关键变量与患病率关系 ==========
def plot_group_rate(df, group_col, label_col, name):
    if group_col not in df.columns or label_col not in df.columns:
        return
    # 分组后患病率
    group = df.groupby(pd.cut(df[group_col].astype(float), bins=[0,30,40,50,60,70,80,120], right=False))[label_col].mean()
    ax = group.plot(kind='bar')
    ax.set_title(f'{name} {group_col}分组后{label_col}均值', fontproperties=font_prop)
    ax.set_ylabel(f'{label_col}均值', fontproperties=font_prop)
    ax.set_xlabel(f'{group_col}分组', fontproperties=font_prop)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop)
    if ax.get_legend() is not None:
        ax.legend(prop=font_prop)
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_OUT_DIR, f'{name}_{group_col}_rate.png'))
    plt.close()

# cirrhosis
if 'Age' in df_cirr.columns and 'Stage' in df_cirr.columns:
    plot_group_rate(df_cirr, 'Age', 'Stage', 'cirrhosis')
# heart
if 'Age' in df_heart.columns and 'HeartDisease' in df_heart.columns:
    plot_group_rate(df_heart, 'Age', 'HeartDisease', 'heart')
# stroke
if 'age' in df_stroke.columns and 'stroke' in df_stroke.columns:
    plot_group_rate(df_stroke, 'age', 'stroke', 'stroke')

# ========== 可视化：箱线图 ==========
def plot_boxplots(df, group_col, name):
    num_cols = df.select_dtypes(include=[np.number]).columns.difference([group_col])
    for col in num_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=group_col, y=col, data=df, palette="Set2", showfliers=True)
        plt.title(f'{name} - {col} 箱线图', fontproperties=font_prop)
        plt.xlabel(group_col, fontproperties=font_prop)
        plt.ylabel(col, fontproperties=font_prop)
        plt.xticks(fontproperties=font_prop)
        plt.yticks(fontproperties=font_prop)
        plt.tight_layout()
        plt.savefig(os.path.join(PLT_OUT_DIR, f'{name}_{col}_boxplot.png'))
        plt.close()

# cirrhosis 按 Stage 分组
if 'Stage' in df_cirr.columns:
    plot_boxplots(df_cirr, 'Stage', 'cirrhosis')
# heart 按 HeartDisease 分组
if 'HeartDisease' in df_heart.columns:
    plot_boxplots(df_heart, 'HeartDisease', 'heart')
# stroke 按 stroke 分组
if 'stroke' in df_stroke.columns:
    plot_boxplots(df_stroke, 'stroke', 'stroke')

print('分析与可视化已完成，结果保存在 output/csv/第一问 和 output/plt/第一问 文件夹。') 