# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:24:34 2019

@author: masoudg
"""

# import some packages
import pandas as pd
import time
import numpy as np
from dateutil.relativedelta import relativedelta
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

#%% data path and files name
data_path         = 'C:/MyFolder/coles_test/data'
result_path       = 'C:/MyFolder/coles_test/results'
train_file_name   = 'train.csv'
store_file_name   = 'stores.csv'
items_file_name   = 'items.csv'
trans_file_name   = 'transactions.csv'
holid_file_name   = 'holidays_events.csv'
oli_file_name     = 'oil.csv'
test_file_name    = 'test.csv' 

#%% Load the data
# As the training data, train.csv is a large file (~5 GB), I set the low_memory=True 
# argument of read_csv, that requires lower memory usage, while at the same time 
# reading the csv's contents into a single DataFrame.
# I also chaged the data types of some columns to make them more memory efficient.
# This is based on information of the data

types_dict = {'id': 'int32',
             'item_nbr': 'int32',
             'store_nbr': 'int8',
             'unit_sales': 'float32'}

# load training data
start_time = time.time()
df_train   = pd.read_csv(data_path + '/' + train_file_name, low_memory=True, dtype=types_dict)
print(f"\n\nLoading {train_file_name} is done! Epalsed time {time.time() - start_time} sec")

# load store data
start_time = time.time()
df_store   = pd.read_csv(data_path + '/' +  store_file_name, low_memory=True)
print(f"Loading {store_file_name} is done! Epalsed time {time.time() - start_time} sec")

# load items data
start_time = time.time()
df_items   = pd.read_csv(data_path + '/' +  items_file_name, low_memory=True)
print(f"Loading {items_file_name} is done! Epalsed time {time.time() - start_time} sec")

# load transaction data
start_time = time.time()
df_trans   = pd.read_csv(data_path + '/' +  trans_file_name, low_memory=True)
print(f"Loading {trans_file_name} is done! Epalsed time {time.time() - start_time} sec")
df_trans['date'] =  pd.to_datetime(df_trans['date'])

# load oil data
start_time = time.time()
df_oil     = pd.read_csv(data_path + '/' +  oli_file_name, low_memory=True)
print(f"Loading {oli_file_name} is done! Epalsed time {time.time() - start_time} sec")
df_oil['date'] =  pd.to_datetime(df_oil['date'])

# load holiday data
start_time = time.time()
df_holid   = pd.read_csv(data_path + '/' +  holid_file_name, low_memory=True)
print(f"Loading {oli_file_name} is done! Epalsed time {time.time() - start_time} sec")
df_holid['date'] =  pd.to_datetime(df_holid['date'])

#%% Some data cleaning and pre-processing
print('start some data cleaning and pre-processing')

# convert the date column to dtype datetime
df_train['date']      = pd.to_datetime(df_train['date'])

# convert FALSE to 0, TRUE to 1, and missing values in onpromotion with -1.
df_train.onpromotion  = df_train.onpromotion.map({False : 0, True : 1})
df_train['onpromotion'].fillna(-1, inplace=True)
df_train.head()

# further update data types of some columns to for more memory efficient.
df_train.onpromotion  = df_train.onpromotion.astype('int8')
df_train.unit_sales   = df_train.unit_sales.astype('float32')  

# Break holidays into separate dataframes
holidays_cities       = df_holid[df_holid.locale == "Local"]       # city level holidays
holidays_states       = df_holid[df_holid.locale == "Regional"]    # state level holidays
holidays_national     = df_holid[df_holid.locale == "National"]  # national holidays
#
# Rename columns to help with joining dataframes later
holidays_cities       = holidays_cities.rename(columns = {'locale_name':'city', 'type':'holiday_type'})
holidays_states       = holidays_states.rename(columns = {'locale_name':'state', 'type':'holiday_type'})
holidays_national     = holidays_national.rename(columns = {'type':'holiday_type'})

# We don't need locale_name at all for national holidays
holidays_national.drop('locale_name', axis=1, inplace=True)

# locale column is useless - let's drop it to simplify joining dataframes
holidays_cities.drop('locale', axis=1, inplace=True)
holidays_states.drop('locale', axis=1, inplace=True)
holidays_national.drop('locale', axis=1, inplace=True)

# transferred column is now useless - let's drop it to simplify joining dataframes
holidays_cities.drop('transferred', axis=1, inplace=True)
holidays_states.drop('transferred', axis=1, inplace=True)
holidays_national.drop('transferred', axis=1, inplace=True)

#%% select a section of training data
print('select a section of training data')

num_months        = 12 
last_dt_train     = df_train.date.max() # Last date available in the training set (train.csv)
# First date of the subset of the training data (1 year prior to test set start)
train_sub_startdt = last_dt_train - relativedelta(months = num_months)
# Sort training data before slicing it based on dates
df_train.sort_values('date', ascending=True, inplace=True)
# We extract data that starts num_months prior to the start of test data
subset_index      = sorted(df_train.index[df_train['date'] == train_sub_startdt].tolist())[0]
train_sub_df      = df_train[subset_index: ].copy()

train_sub_df = pd.merge(train_sub_df, df_trans, how='left', on=['store_nbr', 'date'])
train_sub_df.columns[train_sub_df.isnull().any()]   # these columns have missing values

train_sub_df['was_returned'] = np.where(train_sub_df['unit_sales'] < 0, 1, 0)
train_sub_df.was_returned = train_sub_df.was_returned.astype('int8')

#%% Deal with missing values in transactions and replace them with median
print('Deal with missing values in transactions')

df_trans['year']      = df_trans['date'].dt.year
df_trans['month']     = df_trans['date'].dt.month
#df_trans['day'] = df_trans['date'].dt.day

df_trans_median       = df_trans.groupby(['store_nbr', 'year', 'month'], as_index=False).median()
df_trans_median       = df_trans_median.rename(columns = {'transactions':'transactions_median'})
df_trans.drop(['year', 'month'], axis=1, inplace=True)

# Replace missing transactions with median transactions imputed above
train_sub_df['year']  = train_sub_df['date'].dt.year
train_sub_df['month'] = train_sub_df['date'].dt.month
train_sub_df          = pd.merge(train_sub_df, df_trans_median, how='left', on=['store_nbr', 'year', 'month'])

train_sub_df['transactions'].fillna(train_sub_df['transactions_median'], inplace=True)
train_sub_df.drop('transactions_median', axis=1, inplace=True)
train_sub_df.columns[train_sub_df.isnull().any()]   # these columns have missing values

#%% merging other dataframes

train_sub_df = pd.merge(train_sub_df, df_store, how='left', on='store_nbr')
train_sub_df = pd.merge(train_sub_df, df_items, how='left', on='item_nbr')
train_sub_df = pd.merge(train_sub_df, df_oil, how='left', on=['date'])

train_sub_df.columns[train_sub_df.isnull().any()]   # these columns have missing values


#%% dealing with missing values in oil
print('dealing with missing values in oil')

df_oil['year']  = df_oil['date'].dt.year
df_oil['month'] = df_oil['date'].dt.month

dcoilwtico_mean = df_oil.groupby(['year', 'month'], as_index=False)[['dcoilwtico']].mean()
dcoilwtico_mean = dcoilwtico_mean.rename(columns = {'dcoilwtico':'dcoilwtico_mean'})
dcoilwtico_mean.head(3)

# Replace missing dcoilwtico (oil price) with average dcoilwtico imputed above
train_sub_df    = pd.merge(train_sub_df, dcoilwtico_mean, how='left', on=['year', 'month'])
train_sub_df['dcoilwtico'].fillna(train_sub_df['dcoilwtico_mean'], inplace=True)
train_sub_df.drop('dcoilwtico_mean', axis=1, inplace=True)
train_sub_df.drop(['year', 'month'], axis=1, inplace=True)
df_oil.drop(['year', 'month'], axis=1, inplace=True)

train_sub_df.columns[train_sub_df.isnull().any()]   # these columns have missing values

#%% mergeing holidays
print('adding holidy data')

# city holidays
train_sub_df = pd.merge(train_sub_df, holidays_cities, how='left', on=['date', 'city'])
train_sub_df = train_sub_df.rename(columns = {'holiday_type':'holiday_type_city', 'description':'description_city'})

# state holidays
train_sub_df = pd.merge(train_sub_df, holidays_states, how='left', on=['date', 'state'])
train_sub_df = train_sub_df.rename(columns = {'holiday_type':'holiday_type_state', 'description':'description_state'})

# national holidays
train_sub_df = pd.merge(train_sub_df, holidays_national, how='left', on=['date'])
train_sub_df.rename(columns = {'holiday_type':'holiday_type_nat', 'description':'description_nat'}, inplace=True)

# Impute missing values
train_sub_df['holiday_type_city'].fillna('no holiday', inplace=True)
train_sub_df['holiday_type_state'].fillna('no holiday', inplace=True)
train_sub_df['holiday_type_nat'].fillna('no holiday', inplace=True)
train_sub_df['description_city'].fillna('no holiday', inplace=True)
train_sub_df['description_state'].fillna('no holiday', inplace=True)
train_sub_df['description_nat'].fillna('no holiday', inplace=True)

#%% adding some more feature
print('adding some more feature')

sales_by_store = train_sub_df.groupby(by=['store_nbr'], as_index=False)['unit_sales'].sum()
sales_by_store = sales_by_store.rename(columns = {'unit_sales':'total_store_sales'})
train_sub_df = pd.merge(train_sub_df, sales_by_store, how='left', on=['store_nbr'])


# Number of times every item in every store was returned
num_return_store = train_sub_df.groupby(by='store_nbr', as_index=False)['was_returned'].sum()
num_return_store = num_return_store.rename(columns = {'was_returned':'ret_frq_by_store'})
train_sub_df     = pd.merge(train_sub_df, num_return_store, how='left', on=['store_nbr'])

#Dropping was_returned column
train_sub_df.drop('was_returned', axis=1, inplace=True)

# Change any columns of strings in a panda's dataframe to a column of
# categorical values. 
for n, c in train_sub_df.items():
    if is_string_dtype(c): 
        train_sub_df[n] = c.astype('category').cat.as_ordered()
        
#%% Save the merged train dataframe at this point
print('saving the training data')
train_sub_df.to_feather(data_path + '/train_data')

#%% now time for test data
print('\n\nLoading the test data')
# load training data
start_time = time.time()
test_df    = pd.read_csv(data_path + '/' + test_file_name, low_memory=True, dtype=types_dict)
print(f"\n\nLoading {test_file_name} is done! Epalsed time {time.time() - start_time} sec")

test_df['date'] = pd.to_datetime(test_df['date']);

test_df.onpromotion    = test_df.onpromotion.map({False : 0, True : 1})
test_df['onpromotion'].fillna(-1, inplace=True)
test_df['onpromotion'] = test_df['onpromotion'].astype(np.int8)
test_df.dtypes

# merging some dfs
test_df = pd.merge(test_df, df_store, how='left', on='store_nbr')
test_df = pd.merge(test_df, df_items, how='left', on='item_nbr')
test_df = pd.merge(test_df, df_oil, how='left', on=['date'])

#%% dealing with missing values in oil
print('dealing with missing values in oil and adding it to test data')

test_df['year']  = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month

df_oil['year']  = df_oil['date'].dt.year
df_oil['month'] = df_oil['date'].dt.month

dcoilwtico_mean = df_oil.groupby(['year', 'month'], as_index=False)[['dcoilwtico']].mean()
dcoilwtico_mean = dcoilwtico_mean.rename(columns = {'dcoilwtico':'dcoilwtico_mean'})


# Replace missing dcoilwtico (oil price) with average dcoilwtico imputed above
test_df = pd.merge(test_df, dcoilwtico_mean, how='left', on=['year', 'month'])
test_df['dcoilwtico'].fillna(test_df['dcoilwtico_mean'], inplace=True)
test_df.drop('dcoilwtico_mean', axis=1, inplace=True)
test_df.drop(['year', 'month'], axis=1, inplace=True)
df_oil.drop(['year', 'month'], axis=1, inplace=True)

#%% merge holidays
print('adding holiday data to test data')
test_df = pd.merge(test_df, holidays_cities, how='left', on=['date', 'city'])
test_df = test_df.rename(columns = {'holiday_type':'holiday_type_city', 'description':'description_city'})

test_df = pd.merge(test_df, holidays_states, how='left', on=['date', 'state'])
test_df = test_df.rename(columns = {'holiday_type':'holiday_type_state', 'description':'description_state'})

test_df = pd.merge(test_df, holidays_national, how='left', on=['date'])
test_df.rename(columns = {'holiday_type':'holiday_type_nat', 'description':'description_nat'}, inplace=True)

#% Impute missing values
test_df['holiday_type_city'].fillna('no holiday', inplace=True)
test_df['holiday_type_state'].fillna('no holiday', inplace=True)
test_df['holiday_type_nat'].fillna('no holiday', inplace=True)
test_df['description_city'].fillna('no holiday', inplace=True)
test_df['description_state'].fillna('no holiday', inplace=True)
test_df['description_nat'].fillna('no holiday', inplace=True)

# For every store, add total number of sales
test_df = pd.merge(test_df, sales_by_store, how='left', on=['store_nbr'])

#%% Change any columns of strings in a panda's dataframe to a column of
# categorical values. 
for n, c in test_df.items():
    if is_string_dtype(c): 
        test_df[n] = c.astype('category').cat.as_ordered()
#%%
print('saving the test data')
test_df.to_feather(data_path + '/test_data')

#%% summary
print('some summaries:')

print(train_sub_df.dtypes, '\nn')
print(train_sub_df.head(5))
print(train_sub_df.columns[train_sub_df.isnull().any()])

print(test_df.dtypes, '\nn')
print(test_df.head(5))
print(test_df.columns[test_df.isnull().any()])

























 







    