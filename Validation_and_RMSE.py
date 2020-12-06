#!/usr/bin/env python
# coding: utf-8
import gc

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from joblib import Parallel, delayed


import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



sales_data = pd.read_csv('./data/sales_train_evaluation.csv')
calendar = pd.read_csv('./data/calendar.csv')
price_data = pd.read_csv('./data/sell_prices.csv')

sales_data = reduce_mem_usage(sales_data)
calendar = reduce_mem_usage(calendar)
price_data = reduce_mem_usage(price_data)


#prepare validation data for submission
sales = pd.read_csv('./data/sales_train_evaluation.csv').pipe(reduce_mem_usage, verbose=True)
cols_old = ['id'] + ['d_'+str(x) for x in list(range(1914, 1942))]
cols_new = ['id'] + ['F'+str(x) for x in list(range(1, 29))]

submission_C = sales[cols_old].copy()
submission_C.columns = cols_new
del sales; gc.collect()
submission_C["id"] = submission_C["id"].str.replace("evaluation$", "validation")

#submission_C = pd.read_csv('../input/competition/competition.csv')
#submission_C = submission_C[submission_C.id.str.endswith('validation')]

submission_L = pd.read_csv('submission_validation_Log_price_2020dec15_1.csv')

submission_L = submission_L[submission_L.id.str.endswith('validation')]

NUM_ITEMS = sales_data.shape[0]    # 30490
DAYS_PRED = submission_C.shape[1] - 1    # 28

DAYS_PRED




submission_C.head()





def transform(df):
    newdf = df.melt(id_vars=["id"], var_name="d", value_name="sale")
    newdf.sort_values(by=['id', "d"], inplace=True)
    newdf.reset_index(inplace=True)
    return newdf

from sklearn.metrics import mean_squared_error

def rmse(gt,df):
    gt = transform(gt)
    df = transform(df)
    return np.sqrt(mean_squared_error( gt["sale"],df["sale"]))


# ## Compute RMSE
'''start_date = 1914 
dayCols = ["d_{}".format(i) for i in range(start_date, start_date+DAYS_PRED)]

gt  = sales_data[["id"]+dayCols]
gt.shape
rmse(gt,submission_C)'''

start_date = 1914 
dayCols = ["d_{}".format(i) for i in range(start_date, start_date+DAYS_PRED)]

gt  = sales_data[["id"]+dayCols]
gt.shape
rmse(gt,submission_L)





gt['id2'] = gt['id'].str.replace('evaluation','')
#submission_C['id2'] = submission_C['id'].str.replace('validation','')
submission_L['id2'] = submission_L['id'].str.replace('validation','')

gt = gt[['id','id2'] + dayCols]
FCols = ['F' + str(i) for i in range(1,29)]
#submission_C = submission_C[['id','id2'] + FCols]
submission_L = submission_L[['id','id2'] + FCols]




def rmse2(submission,i):
    SE_col =[]
    for j in range(2,submission.shape[1]):
        y = float(gt.iloc[i,j])
        yhat = float(submission.iloc[i,j])
        #print(y,yhat)
        e_i = (y-yhat)**2
        SE_col.append(e_i)
    #print(len(SE_col))
    #print(gt.iloc[i,0])
    return np.sqrt(sum(SE_col)/len(SE_col))




#RMSE_id_C = Parallel(n_jobs=4, backend="multiprocessing")(delayed(rmse2)(submission_C,i) for i in  range(0,submission_C.shape[0]))
#np.mean(RMSE_id_C)



#len(RMSE_id_C)



RMSE_id_L = Parallel(n_jobs=4, backend="multiprocessing")(delayed(rmse2)(submission_L,i) for i in  range(0,submission_L.shape[0]))
np.mean(RMSE_id_L)




import seaborn as sns
#sns.boxplot(RMSE_id_C)




df_RMSE = pd.DataFrame()
#df_RMSE['RMSE_Model1']  = np.array(RMSE_id_C)
df_RMSE['RMSE_Model2']  = np.array(RMSE_id_L)




import matplotlib.pyplot as plt
#log_RMSE_id_C = [np.log(i) for i in RMSE_id_C]
plt.figure(figsize = (5,5))
df_RMSE.plot()
plt.xlabel('id _no')
plt.ylabel('RMSE')
plt.title('RMSE - Sequence- Model 1 and Model2')




import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,5))
#lst1 = list(df_RMSE['RMSE_Model1'])
lst2 = list(df_RMSE['RMSE_Model2'])
#sns.distplot(lst1,kde_kws={"alpha":0.5},hist_kws={"alpha":0.5})
sns.distplot(lst2,kde_kws={"alpha":0.5},hist_kws={"alpha":0.5})
fig.legend(labels=['Model1','Model2'])
plt.xlabel('RMSE')
plt.ylabel('density')
plt.title('RMSE - Sequence- Model 1 and Model2')
plt.show()
#print('N_ items Model 1',df_RMSE['RMSE_Model1'].shape[0],'N_ items Model 2',df_RMSE['RMSE_Model2'].shape[0])




df_RMSE.shape



import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,5))
#lst1 = list(df_RMSE[df_RMSE['RMSE_Model1']<10]['RMSE_Model1'])
lst2 = list(df_RMSE[df_RMSE['RMSE_Model2']<10]['RMSE_Model2'])
#sns.distplot(lst1,kde_kws={"alpha":0.5},hist_kws={"alpha":0.5})
sns.distplot(lst2,kde_kws={"alpha":0.5},hist_kws={"alpha":0.5})
fig.legend(labels=['Model1','Model2'])
plt.xlabel('RMSE')
plt.ylabel('density')
plt.title('RMSE - Sequence- Model 1 and Model2')
plt.show()
#print('N_ items Model 1',df_RMSE[df_RMSE['RMSE_Model1']<10]['RMSE_Model1'].shape[0],'N_ items Model 2',df_RMSE[df_RMSE['RMSE_Model2']<10]['RMSE_Model2'].shape[0])


# ## Compute RMSSE



def comp_rmsse(submission,counter):
    sales_data.head()
    train_column = sales_data.iloc[:,6:1919].columns.tolist()
    dif1 = []
    for i in range(6,1919):
        if i>6:
            temp = (sales_data.iloc[counter,i]  - sales_data.iloc[counter,i-1])**2
            dif1.append(temp)
    den = np.sqrt(sum(dif1)/(len(dif1)-1))
    #print(len(dif1))
    num = rmse2(submission,counter)
    rmsse = num/den
    return rmsse




#RMSSE_C = Parallel(n_jobs=4, backend="multiprocessing")(delayed(comp_rmsse)(submission_C,i) for i in  range(0,sales_data.shape[0]))
#np.mean(RMSSE_C)



#RMSSE_C[25]



RMSSE_L = Parallel(n_jobs=4, backend="multiprocessing")(delayed(comp_rmsse)(submission_L,i) for i in  range(0,sales_data.shape[0]))
np.mean(RMSSE_L)



RMSSE = pd.DataFrame()
#RMSSE['RMSSE_Model1']  = np.array(RMSSE_C)
RMSSE['RMSSE_Model2']  = np.array(RMSSE_L)



import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,5))
#lst1 = list(RMSSE.loc[RMSSE['RMSSE_Model1']<10]['RMSSE_Model1'])
lst2 = list(RMSSE.loc[RMSSE['RMSSE_Model1']<10]['RMSSE_Model2'])
#sns.distplot(lst1,kde_kws={"alpha":0.5},hist_kws={"alpha":0.5})
sns.distplot(lst2,kde_kws={"alpha":0.5},hist_kws={"alpha":0.5})
fig.legend(labels=['Model1','Model2'])
plt.xlabel('RMSSE')
plt.ylabel('density')
plt.title('RMSSE - Sequence- Model 1 and Model2')
plt.show()
#print('N_ items Model 1',RMSSE['RMSSE_Model1'].shape[0],'N_ items Model 2',RMSSE['RMSSE_Model2'].shape[0])

