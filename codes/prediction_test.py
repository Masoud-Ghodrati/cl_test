# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:58:37 2019

@author: masoudg
"""

# import some packages
import pandas as pd
import os
import re
import numpy as np
from dateutil.relativedelta import relativedelta
import convertNumeric
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import forest
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

#%% data path and files name
data_path         = 'C:/MyFolder/coles_test/data'
result_path       = 'C:/MyFolder/coles_test/results'

#%% load the train and test data
print('loading the train and test data')
df_train = pd.read_feather(data_path +'/train_data')
df_train.drop(['transactions', 'ret_frq_by_store'], axis = 1, inplace = True)
df_test  = pd.read_feather(data_path +'/test_data')
#%%
## Not setting one-hot encoding -> max_n_cat = 7
#cat_columns           = df_train.select_dtypes(['category']).columns
#df_train[cat_columns] = df_train[cat_columns].apply(lambda x: x.cat.codes)
#df_trn, y_trn         = df_train.loc[:, df_train.columns != 'unit_sales'], np.ravel(df_train.loc[:, df_train.columns == 'unit_sales'])
#
### add dummy column
#df_test['dummy']      = 1
#cat_columns           = df_test.select_dtypes(['category']).columns
#df_test[cat_columns]  = df_test[cat_columns].apply(lambda x: x.cat.codes)                

df_trn, y_trn, _ = convertNumeric.proc_df(df_train, 'unit_sales', ignore_flds='date')
y_trn            = np.log1p(np.clip(y_trn, 0, None))

# add dummy column
df_test['dummy'] = 1
df_ts, _, _     = convertNumeric.proc_df(df_test, 'dummy', ignore_flds='date')

#%% 

def split_vals(sub_df, trn_ind, val_ind): 
    return sub_df[0:trn_ind].copy(), sub_df[trn_ind:val_ind].copy()

# Perform k-fold cross-validation with Random Forest parameters
def rf_reg_crossval(df, y, df_ts, n_months, n_days, n_est, min_sam_leaf):
    
    n_folds        = int(n_months*30.0/n_days)
    errors         = []
    fis            = []
    first_dt_train = df.date.min() # First date available in the training set (train.csv)
    
    #Setting seed
    np.random.seed(9001)
    
    #Sequentially split available training dataset into train and validation sets
    for i in range(1, n_folds):
        print('*____________________________________*')
        print('running fold # ', i, ' of ', n_folds)

        
        forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, 1000000))
                
        # Initialize random forest
        m_kcv = RandomForestRegressor(n_estimators = n_est, max_features = 0.5,
                                      min_samples_leaf = min_sam_leaf, n_jobs=-1, 
                                      oob_score = False)
       
        
        # Getting dates for training and validation sets
        train_sub_startdt = first_dt_train + relativedelta(days = i*n_days)
        valid_sub_startdt = first_dt_train + relativedelta(days = (i+1)*n_days)   
        
        # Create indices for training and validation sets
        index_train = sorted(df.index[df['date'] == train_sub_startdt].tolist())[0]
        index_valid = sorted(df.index[df['date'] == valid_sub_startdt].tolist())[0]
        
        # Create subsetted dataframes of X,y, and w for training and validation sets
        X_train, X_valid = split_vals(df.loc[:, df.columns != 'date'], index_train, index_valid)
        y_train, y_valid = split_vals(y, index_train, index_valid)
                
        print('X train shape:      ', X_train.shape)
        print('Y train shape:      ', y_train.shape)
        print('X validation shape: ', X_valid.shape)
        print('Y validation shape: ', y_valid.shape)
        
        # Compute arrays of item score weights for the items in the validation
        # set (for which we will make predictions)
        item_weight_train = 1 + X_train['perishable'] * 0.25
        item_weight_valid = 1 + X_valid['perishable'] * 0.25
        
        # Optimizing the model fit by converting it to float array outside
        X_train= np.array(X_train, dtype=np.float32)
        
        # Fit the random forest model on training set w/ cross validation
        m_kcv.fit(X_train, y_train)

        # Print the NWRMSLE score and R-squared values for training and validation sets
        train_score = prediction_score(m_kcv, X_train, y_train, item_weight_train, plot_pre=False)
        val_score   = prediction_score(m_kcv, X_valid, y_valid, item_weight_valid, plot_pre=False)
        print('For training set:   [nwrmsle, rsquared]: ', train_score)
        print('For validation set: [nwrmsle, rsquared]: ', val_score)
        print('\n')
        
        # Add errors to the errors list
        errors.append([train_score, val_score])  
        
        # Feature importance
        fi = pd.DataFrame({'col_names':df.loc[:, df.columns != 'date'].columns, 
                           'feature_imp': m_kcv.feature_importances_}).sort_values('feature_imp', ascending=False)       
        print(fi[:10])
        fis.append(fi[:10])
        
        # Reducing number of features
        to_keep = fi[fi.feature_imp> 0.005].col_names; # we nedd to keep perishable for scoring
        df_keep = df[to_keep].copy()
        
        # Training on training set again
        # Create subsetted dataframes of X,y, and w for training and validation sets
        X_train, X_valid = split_vals(df_keep.loc[:, df_keep.columns != 'date'], index_train, index_valid)
        y_train, y_valid = split_vals(y, index_train, index_valid)
        
        print('\nTraining again on selected features')
        print('X train shape:      ', X_train.shape)
        print('Y train shape:      ', y_train.shape)
        print('X validation shape: ', X_valid.shape)
        print('Y validation shape: ', y_valid.shape)
        
        # Compute arrays of item score weights for the items in the validation set (for which we will make predictions)
        item_weight_train = 1 + X_train['perishable'] * 0.25
        item_weight_valid = 1 + X_valid['perishable'] * 0.25
        
        # Optimizing the model fit by converting it to float array outside
        X_train= np.array(X_train, dtype=np.float32)
        
        # Fit the random forest model on training set w/ cross validation
        m_kcv.fit(X_train, y_train)
        
        # Print the NWRMSLE score and R-squared values for training and validation sets
        train_score = prediction_score(m_kcv, X_train, y_train, item_weight_train, plot_pre=False)
        val_score   = prediction_score(m_kcv, X_valid, y_valid, item_weight_valid, plot_pre=True)
        print('For training set after feature selection  : [nwrmsle, rsquared]: ', train_score)
        print('For validation set after feature selection: [nwrmsle, rsquared]: ', val_score)
        
        # Add errors to the errors list
        errors.append([train_score, val_score])  
        
        # Train on entire training set
        # Initialize random forest
        m_kcv_new   = RandomForestRegressor(n_estimators = n_est, max_features = 0.5,
                                          min_samples_leaf = min_sam_leaf, n_jobs=-1, oob_score = False)
        
        X_train_new = df_keep.loc[:, df_keep.columns != 'date'][ : index_valid]
        y_train_new = y[:index_valid]
               
        # Compute arrays of item score weights for the items in the training 
        # set (for which we will make predictions)
        item_weight_train_new = 1 + X_train_new['perishable'] * 0.25
        
        # Optimizing the model fit by converting it to float array outside
        X_train_new = np.array(X_train_new, dtype=np.float32)
        
        print('\nTraining again on whole training set with selected features')
        print('X train shape:      ', X_train_new.shape)
        print('Y train shape:      ', y_train_new.shape)
        
        # Fit the random forest model on training set w/ cross validation
        m_kcv_new.fit(X_train_new, y_train_new)
        
        # Print the NWRMSLE score and R-squared values for training and validation sets
        train_score = prediction_score(m_kcv_new, X_train_new, y_train_new, item_weight_train_new, plot_pre=False)
        print('For whole training set after feature selection  : [nwrmsle, rsquared]: ', train_score)
        
        
#        #  test set
#        test_keep     = df_ts[to_keep].copy()
#        
#        print('\nTesting on test data')
#        #Predict for test set
#        pred_test_log = m_kcv_new.predict(test_keep)
#        pred_test     = np.round(np.expm1(pred_test_log), decimals=0)
#        output        = pd.concat([test_keep['id'], pd.DataFrame(pred_test)], axis=1)
#        output.columns = ['id', 'unit_sales']
#        name           = 'prediction' + str(i) + '.csv' 
#        output.to_csv(name, index=False)
        
    # Write errors to file
    with open('errors_rf_reg.txt', 'w') as file:
        file.write(str(errors))
    
    # Write feature importances to file
    with open('feature_importance_rf_reg.txt', 'w') as file:
        file.write(str(fis))
 

# Perform k-fold cross-validation with Random Forest parameters
def lin_reg_crossval(df, y, df_ts, n_months, n_days):
    
    n_folds        = int(n_months*30.0/n_days)
    errors         = []
    fis            = []
    first_dt_train = df.date.min() # First date available in the training set (train.csv)
    
    #Setting seed
    np.random.seed(9001)
    
    #Sequentially split available training dataset into train and validation sets
    for i in range(1, n_folds):
        print('*____________________________________*')
        print('running fold # ', i, ' of ', n_folds)
      
        # Getting dates for training and validation sets
        train_sub_startdt = first_dt_train + relativedelta(days = i*n_days)
        valid_sub_startdt = first_dt_train + relativedelta(days = (i+1)*n_days)   
        
        # Create indices for training and validation sets
        index_train = sorted(df.index[df['date'] == train_sub_startdt].tolist())[0]
        index_valid = sorted(df.index[df['date'] == valid_sub_startdt].tolist())[0]
        
        # Create subsetted dataframes of X,y, and w for training and validation sets
        X_train, X_valid = split_vals(df.loc[:, df.columns != 'date'], index_train, index_valid)
        y_train, y_valid = split_vals(y, index_train, index_valid)
                
        print('X train shape:      ', X_train.shape)
        print('Y train shape:      ', y_train.shape)
        print('X validation shape: ', X_valid.shape)
        print('Y validation shape: ', y_valid.shape)
        
        # Compute arrays of item score weights for the items in the validation
        # set (for which we will make predictions)
        item_weight_train = 1 + X_train['perishable'] * 0.25
        item_weight_valid = 1 + X_valid['perishable'] * 0.25
        
        # Optimizing the model fit by converting it to float array outside
        X_train= np.array(X_train, dtype=np.float32)
        
        # Fit the linear reg model on training set w/ cross validation
        m_lr = LinearRegression()
        m_lr.fit(X_train, y_train)
        # Print the NWRMSLE score and R-squared values for training and validation sets
        train_score = prediction_score(m_lr, X_train, y_train, item_weight_train, plot_pre=False)
        val_score   = prediction_score(m_lr, X_valid, y_valid, item_weight_valid, plot_pre=False)
        print('For training set:   [nwrmsle, rsquared]: ', train_score)
        print('For validation set: [nwrmsle, rsquared]: ', val_score)
        print('\n')
        
        # Add errors to the errors list
        errors.append([train_score, val_score])

        # Feature importance
        
        selector = RFE(m_lr, 10, step=1)
        selector = selector.fit(X_train, y_train)
        
        fi = pd.DataFrame({'col_names':df.loc[:, df.columns != 'date'].columns, 
                           'feature_imp': selector.support_})      
        print(fi[fi.feature_imp==True])
        fis.append(fi[fi.feature_imp==True])

        # Reducing number of features
        to_keep = fi[fi.feature_imp==True].col_names; # we nedd to keep perishable for scoring
        df_keep = df[to_keep].copy()
        
                # Training on training set again
        # Create subsetted dataframes of X,y, and w for training and validation sets
        X_train, X_valid = split_vals(df_keep.loc[:, df_keep.columns != 'date'], index_train, index_valid)
        y_train, y_valid = split_vals(y, index_train, index_valid)
        
        print('\nTraining again on selected features')
        print('X train shape:      ', X_train.shape)
        print('Y train shape:      ', y_train.shape)
        print('X validation shape: ', X_valid.shape)
        print('Y validation shape: ', y_valid.shape)
        
        # Compute arrays of item score weights for the items in the validation set (for which we will make predictions)
        item_weight_train = 1 + X_train['perishable'] * 0.25
        item_weight_valid = 1 + X_valid['perishable'] * 0.25
        
        # Optimizing the model fit by converting it to float array outside
        X_train= np.array(X_train, dtype=np.float32)
        
        # Fit the random forest model on training set w/ cross validation
        m_lr.fit(X_train, y_train)
        
        # Print the NWRMSLE score and R-squared values for training and validation sets
        train_score = prediction_score(m_lr, X_train, y_train, item_weight_train, plot_pre=False)
        val_score   = prediction_score(m_lr, X_valid, y_valid, item_weight_valid, plot_pre=True)
        print('For training set after feature selection  : [nwrmsle, rsquared]: ', train_score)
        print('For validation set after feature selection: [nwrmsle, rsquared]: ', val_score)
        
        # Add errors to the errors list
        errors.append([train_score, val_score])  

        # Train on entire training set  
        m_lr_new    = LinearRegression()
        
        X_train_new = df_keep.loc[:, df_keep.columns != 'date'][ : index_valid]
        y_train_new = y[:index_valid]
               
        # Compute arrays of item score weights for the items in the training 
        # set (for which we will make predictions)
        item_weight_train_new = 1 + X_train_new['perishable'] * 0.25
        
        # Optimizing the model  by converting it to float array outside
        X_train_new = np.array(X_train_new, dtype=np.float32)
        
        print('\nTraining again on whole training set with selected features')
        print('X train shape:      ', X_train_new.shape)
        print('Y train shape:      ', y_train_new.shape)
        
        # Fit the random forest model on training set w/ cross validation
        m_lr_new.fit(X_train_new, y_train_new)
        
        # Print the NWRMSLE score and R-squared values for training and validation sets
        train_score = prediction_score(m_lr_new, X_train_new, y_train_new, item_weight_train_new, plot_pre=False)
        print('For whole training set after feature selection  : [nwrmsle, rsquared]: ', train_score)    
        
        # test set
#        test_keep     = df_ts[to_keep].copy()
#        
#        print('\nTesting on test data')
#        #Predict for test set
#        pred_test_log  = m_lr_new.predict(test_keep)
#        pred_test      = np.round(np.expm1(pred_test_log), decimals=0)
#        output         = pd.concat([df_ts['id'], pd.DataFrame(pred_test)], axis=1)
#        output.columns = ['id', 'unit_sales']
#        name           = 'prediction' + str(i) + '.csv' 
#        output.to_csv(name, index=False)
        
    # Write errors to file
    with open('errors_lin_reg.txt', 'w') as file:
        file.write(str(errors))
    
    # Write feature importances to file
    with open('feature_importance_lin_reg.txt', 'w') as file:
        file.write(str(fis))
       
# Computes and returns NWRMSLE score and R-squared values
def prediction_score(model, X, y, weights, plot_pre=False):
    # Predicting for the input data
    y_hat    = model.predict(X)
    # 
    if plot_pre:
        plot_predVture(y_hat, y, result_path, model)
    # Calculating the residuals
    rsquared = model.score(X, y)
    
    # Specified score (from Kaggle)
    nwrmsle = np.sqrt(np.sum(np.multiply(weights, np.square(y_hat - y))) / np.sum(weights))
    res = [nwrmsle, rsquared]
    if hasattr(model, 'oob_score_'): 
        res.append(model.oob_score_)
    return res

def plot_predVture(y_hat, y, result_path, model):
    
    fig = plt.figure()
    fig.set_size_inches(4, 2)
    plt.plot(y_hat[:70], 'r', label='prediction')
    plt.plot(y[:70], 'b', label='true')
    plt.ylabel('Unit sales')
    plt.xlabel('Items')
    plt.box(on=False)
    
    file_list = os.listdir(result_path + '/')
    
    if len(file_list)==0:
        file_name = type(model).__name__ + str(1) + '.png'
    else:
        f_num = []
        for fn in file_list:
            f_num.append(int(re.findall('\d+', fn )[0]));
        
        file_name = type(model).__name__ + str(max(f_num)+1) + '.png'
        
    fig.savefig(result_path + '/' + file_name, dpi=300)    
    
#%% run the models
    
rf_reg_crossval(df_trn, y_trn, df_ts, n_months=10, n_days=15, n_est = 80, min_sam_leaf = 3)    
lin_reg_crossval(df_trn, y_trn, df_ts, n_months=10, n_days=15)    

