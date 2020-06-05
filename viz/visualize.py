# Title     : visualization module
# Objective : framework for generalized visualization
# Created by: abhi
# Created on: 6/4/20

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings('ignore')


def density_plot(data_frame, column):
    sns.distplot(data_frame[column])
    plt.show()
    # skewness and kurtosis
    print("Skewness: %f" % data_frame[column].skew())
    print("Kurtosis: %f" % data_frame[column].kurt())


def scatter_plot(data_frame, dependent, column):
    data = pd.concat([data_frame[dependent], data_frame[column]], axis=1)
    data.plot.scatter(x=column, y=dependent, ylim=(0, 800000))
    plt.show()


def box_plot(data_frame, dependent, column):
    data = pd.concat([data_frame[dependent], data_frame[column]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=column, y=dependent, data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()


def corr_mat(data_frame):
    corrmat = data_frame.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.show()


def corr_mat_topk(data_frame, dependent, k):
    corrmat = data_frame.corr()
    cols = corrmat.nlargest(k, dependent)[dependent].index
    cm = np.corrcoef(data_frame[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


def pair_plot(data_frame, columns):
    sns.set()
    sns.pairplot(data_frame[columns], size=2.5)
    plt.show()


def hist_normal_prob(data_frame, dependent):
    sns.distplot(data_frame[dependent], fit=norm);
    fig = plt.figure()
    res = stats.probplot(data_frame[dependent], plot=plt)
    plt.show()


def missing_data_plot(data_frame):
    data_frame_na = (data_frame.isnull().sum() / len(data_frame)) * 100
    data_frame_na = data_frame_na.drop(data_frame_na[data_frame_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': data_frame_na})
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=data_frame_na.index, y=data_frame_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()

def target_distribution(data_frame, dependent, cont_covariate):
    data_frame['TransactionAmt'] = data_frame[cont_covariate].astype(float)
    total = len(data_frame)
    total_amt = data_frame.groupby([dependent])[cont_covariate].sum().sum()
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    g = sns.countplot(x=dependent, data=data_frame, )
    g.set_title("Fraud Transactions Distribution \n# 0: No Fraud | 1: Fraud #", fontsize=22)
    g.set_xlabel("Is fraud?", fontsize=18)
    g.set_ylabel('Count', fontsize=18)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=15)
    perc_amt = (data_frame.groupby([dependent])[cont_covariate].sum())
    perc_amt = perc_amt.reset_index()
    plt.subplot(122)
    g1 = sns.barplot(x=dependent, y=cont_covariate,  dodge=True, data=perc_amt)
    g1.set_title("% Total Amount in Transaction Amt \n# 0: No Fraud | 1: Fraud #", fontsize=22)
    g1.set_xlabel("Is fraud?", fontsize=18)
    g1.set_ylabel('Total Transaction Amount Scalar', fontsize=18)
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt * 100),
                ha="center", fontsize=15)
    plt.show()

target_distribution(df_trans,'isFraud', 'TransactionAmt')

def cat_var_dep_cont_indep(data_frame, dependent, cat_var, cont_var):
    total = len(df_trans)
    tmp = pd.crosstab(data_frame[cat_var], data_frame[dependent], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    plt.figure(figsize=(14,10))
    plt.suptitle('ProductCD Distributions', fontsize=22)
    plt.subplot(221)
    g = sns.countplot(x=cat_var, data=df_trans)
    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])
    g.set_title("ProductCD Distribution", fontsize=19)
    g.set_xlabel("ProductCD Name", fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    g.set_ylim(0,500000)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14)
    plt.subplot(222)
    g1 = sns.countplot(x=cat_var, hue=dependent, data=data_frame)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x=cat_var, y='Fraud', data=tmp, color='black', order=['W', 'H',"C", "S", "R"], legend=False) # change order as per level in the cat_var
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)
    g1.set_title("Product CD by Target(isFraud)", fontsize=19)
    g1.set_xlabel("ProductCD Name", fontsize=17)
    g1.set_ylabel("Count", fontsize=17)
    plt.subplot(212)
    g3 = sns.boxenplot(x=cat_var, y=cont_var, hue=dependent,
                  data=data_frame[data_frame[cont_var] <= 2000] ) # reomove filter as per the need
    g3.set_title("Transaction Amount Distribuition by ProductCD and Target", fontsize=20)
    g3.set_xlabel("ProductCD Name", fontsize=17)
    g3.set_ylabel("Transaction Values", fontsize=17)
    plt.subplots_adjust(hspace = 0.6, top = 0.85)
    plt.show()

cat_var_dep_cont_indep(df_trans,'isFraud','ProductCD','TransactionAmt')

def dep_cont_var_distribution(data_frame, dependent, cont_var):
    g = sns.distplot(data_frame[data_frame[dependent] == 1][cont_var], label='Fraud')
    g = sns.distplot(data_frame[data_frame[dependent] == 0][cont_var], label='NoFraud')
    g.legend()
    g.set_title(f"{cont_var} Values Distribution by Target", fontsize=20)
    g.set_xlabel(f"{cont_var} Values", fontsize=18)
    g.set_ylabel("Probability", fontsize=18)

dep_cont_var_distribution(df_trans,'isFraud','card1' )

df_trans.loc[df_trans.addr1.isin(df_trans.addr1.value_counts()[df_trans.addr1.value_counts() <= 5000 ].index), 'addr1'] = "Others"
def plotting_cnt_amt(data_frame, cat_var, dependent, cnt_var, lim=2000):
    tmp = pd.crosstab(data_frame[cat_var], data_frame[dependent], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(16,14))
    plt.suptitle(f'{cat_var} Distributions ', fontsize=24)

    plt.subplot(211)
    g = sns.countplot( x=cat_var,  data=df, order=list(tmp[cat_var].values))
    gt = g.twinx()
    gt = sns.pointplot(x=cat_var, y='Fraud', data=tmp, order=list(tmp[cat_var].values),
                       color='black', legend=False, )
    gt.set_ylim(0,tmp['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {cat_var} values and % {dependent} Transactions", fontsize=20)
    g.set_xlabel(f"{cat_var} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=12)

    g.set_ylim(0,max(sizes)*1.15)

    #########################################################################
    perc_amt = (data_frame.groupby([dependent,cat_var])[cnt_var].sum() \
                / data_frame.groupby([cat_var])[cnt_var].sum() * 100).unstack(dependent)
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    amt = df.groupby([cat_var])[cnt_var].sum().reset_index()
    perc_amt = perc_amt.fillna(0)
    plt.subplot(212)
    g1 = sns.barplot(x=cat_var, y=cnt_var,
                       data=amt,
                       order=list(tmp[cat_var].values))
    g1t = g1.twinx()
    g1t = sns.pointplot(x=cat_var, y='Fraud', data=perc_amt,
                        order=list(tmp[cat_var].values),
                       color='black', legend=False, )
    g1t.set_ylim(0,perc_amt['Fraud'].max()*1.1)
    g1t.set_ylabel("%{cat_var} Total {cnt_var}", fontsize=16)
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    g1.set_title(f"{cat_var} by {cnt_var} Total + %of total and %{cat_var} Transactions", fontsize=20)
    g1.set_xlabel(f"{cat_var} Category Names", fontsize=16)
    g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)
    g1.set_xticklabels(g.get_xticklabels(),rotation=45)

    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt*100),
                ha="center",fontsize=12)

    plt.subplots_adjust(hspace=.4, top = 0.9)
    plt.show()

plotting_cnt_amt(df_trans, 'addr1', 'isFraud', 'TransactionAmt' )



if __name__ == "__main__":
    df = pd.read_csv("../input/train_house_price.csv")
    # density_plot(df, 'SalePrice')
    # scatter_plot(df, 'SalePrice', 'GrLivArea')
    # box_plot(df, 'SalePrice', 'OverallQual')
    # corr_mat(df)
    # corr_mat_topk(df, 'SalePrice', 10)
    # pair_plot(df, ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'])
    #hist_normal_prob(df, 'SalePrice')
    missing_data_plot(df)
