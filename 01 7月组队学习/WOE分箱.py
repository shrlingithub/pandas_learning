# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 15:31:22 2022

@author: shrlin
"""

import pandas as pd
import numpy as np
df = pd.read_csv('train.csv',names=['乘客ID','target','乘客等级(1/2/3等舱位)','乘客姓名','性别','年龄',
                                      '堂兄弟/妹个数','父母与小孩个数','船票信息','票价','客舱','登船港口'],
                               index_col = '乘客ID',
                               header = 0)

list(np.unique(df['乘客等级(1/2/3等舱位)']))


def calulate_iv(df,var,global_bt,global_gt):
    '''
    calculate the iv and woe value without split
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :return:
    '''
    # a = df.groupby(['target']).count()
    groupdetail = {}
    bt_sub = sum(df['target'])
    bri = (bt_sub + 0.0001)* 1.0 / global_bt
    gt_sub = df.shape[0] - bt_sub
    gri = (gt_sub + 0.0001)* 1.0 / global_gt

    groupdetail['woei'] = np.log(bri / gri)
    groupdetail['ivi'] = (bri - gri) * np.log(bri / gri)
    groupdetail['sub_total_num_percentage'] = df.shape[0]*1.0/(global_bt+global_gt)
    groupdetail['positive_sample_num'] = bt_sub
    groupdetail['negative_sample_num'] = gt_sub
    groupdetail['positive_rate_in_sub_total'] = bt_sub*1.0/df.shape[0]
    groupdetail['negative_rate_in_sub_total'] = gt_sub*1.0/df.shape[0]

    return groupdetail

groupdetail = calulate_iv(df,'票价',sum(df['target']),df.shape[0] - sum(df['target']))

def calculate_iv_split(df,var,split_point,global_bt,global_gt):
    """
    calculate the iv value with the specified split point
    note:
        the dataset should have variables:'target' which to be encapsulated if have time
    :return:
    """
    #split dataset
    dataset_r = df[df.loc[:,var] > split_point][[var,'target']]
    dataset_l = df[df.loc[:,var] <= split_point][[var,'target']]

    r1_cnt = sum(dataset_r['target'])
    r0_cnt = dataset_r.shape[0] - r1_cnt

    l1_cnt = sum(dataset_l['target'])
    l0_cnt = dataset_l.shape[0] - l1_cnt

    if r0_cnt == 0 or r1_cnt == 0 or l0_cnt == 0 or l1_cnt ==0:
        return 0,0,0,dataset_l,dataset_r,0,0

    lbr = (l1_cnt+ 0.0001)*1.0/global_bt
    lgr = (l0_cnt+ 0.0001)*1.0/global_gt
    woel = np.log(lbr/lgr)
    ivl = (lbr-lgr)*woel
    rbr = (r1_cnt+ 0.0001)*1.0/global_bt
    rgr = (r0_cnt+ 0.0001)*1.0/global_gt
    woer = np.log(rbr/rgr)
    ivr = (rbr-rgr)*woer
    iv = ivl+ivr

    return woel,woer,iv,dataset_l,dataset_r,ivl,ivr

woel,woer,iv,dataset_l,dataset_r,ivl,ivr = calculate_iv_split(df,'票价',10,sum(df['target']),df.shape[0] - sum(df['target']))




'''
pandas : 1.3.4
numpy : 1.20.3
scipy : 1.7.1
re : 2.2.1
'''
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


max_bin = 20
force_bin = 3

# define a binning function
def mono_bin(Y, X, method = 'dp',n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    # 这里要求分箱的迭代结果是完全线性相关的，可能存在优化的机会
    while np.abs(r) < 1:
        try:
            '''
                # 等频分箱 pd.qcut(value_list, q = n)
                # 等距分箱 pd.cut(value_list, bins = n)
            '''
            if method == 'dp':
                d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            elif method == 'dj':
                d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, n)})
            
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df, target,method = 'dp'):
    x = df.dtypes.index.to_list()
    x.remove(target.name)
    count = -1
    
    for i in x :
        if np.issubdtype(df[i], np.number) and len(Series.unique(df[i])) > 2:
            print(f'****** {i}被识别为连续型变量 ******')
            conv = mono_bin(target, df[i],method = method)
            conv["VAR_NAME"] = i
            count = count + 1
        else:
            print(f'****** {i}被识别为分类型变量 ******')
            conv = char_bin(target, df[i])
            conv["VAR_NAME"] = i     
            count = count + 1
            
        x_label = conv.MIN_VALUE.apply(lambda x:str(x)) + '-' + conv.MAX_VALUE.apply(lambda x:str(x))
        plt.bar(x_label, conv['WOE'])
        plt.title(i)
        plt.show()
        
        if count == 0:
            iv_df = conv
        else:
            iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv) 

# df_drop = df.drop(['乘客姓名','船票信息'],axis = 1)
df = df.drop(['乘客姓名','船票信息'],axis = 1)
df['客舱'] = df['客舱'].apply(lambda x: str(x)[:1] if not pd.isna(x) else x)
df['乘客等级(1/2/3等舱位)'] = df['乘客等级(1/2/3等舱位)'].apply(lambda x: str(x) if not pd.isna(x) else x)
final_iv, IV = data_vars(df[['堂兄弟/妹个数','是否幸存']],df.是否幸存,method = 'dj')




import pandas as pd

df = pd.read_csv('train.csv',names=['乘客ID','是否幸存','乘客等级(1/2/3等舱位)','乘客姓名','性别','年龄',
                                      '堂兄弟/妹个数','父母与小孩个数','船票信息','票价','客舱','登船港口'],
                               index_col = '乘客ID',
                               header = 0)
df1 = pd.DataFrame({"X": df['堂兄弟/妹个数'], "Y": df['是否幸存']})
justmiss = df1[['X','Y']][df1.X.isnull()]
notmiss = df1[['X','Y']][df1.X.notnull()]


'''
等频分箱
'''
d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, 20,duplicates = 'drop')})
d2 = d1.groupby('Bucket', as_index=True)
d2.mean()


'''
等深分箱
'''
d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, 20,duplicates = 'drop')})
d2 = d1.groupby('Bucket', as_index=True)
d2.mean()

