# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Regressors
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import LinearRegression, ElasticNet 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler,
                                   RobustScaler, PolynomialFeatures)
from sklearn.neighbors import KNeighborsRegressor as KNN

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer, KNNImputer

import zipfile
import missingno as msno

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#%% #* functions
def plot_comparison(y, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([15, 200], [15, 200], c='k', ls='--')
    ax.set_xlim((15, 200))
    ax.set_ylim((15, 200))
    ax.set_xlabel('True Temperature, C')
    ax.set_ylabel('Estimated True Temperature, C')
    ax.set_aspect('equal', 'box')

    return fig

def plot_comparison_HG(df, dfhg):
    df2 = df.merge(dfhg, on='UWI', how='left')
    return plot_comparison(df2['TrueTemp_x'], df2['TrueTemp_y'])

#%% #* Load data
DV = pd.read_csv("../data/Duvernay_merged.csv")
EB = pd.read_csv("../data/Eaglebine_merged.csv")

best = pd.read_csv('../best prediction/predictions.csv')

best['UWI'] = best['UWI'].astype(str)
DV['UWI'] = DV['UWI'].astype(str)
EB['UWI'] = EB['UWI'].astype(str)

# %% #* Define model
def linear_model(X, y):
    scalers = [MaxAbsScaler(), RobustScaler(), MinMaxScaler(), StandardScaler()]

    pipe = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('pca', PCA()),
        ('poly', PolynomialFeatures()),
        ('bag', BaggingRegressor(base_estimator=LinearRegression(), 
                                    n_estimators=100, n_jobs=-1, random_state=7)),
    ])

    params = {
        'scaler': scalers,
        'pca__n_components': [2, 3, 4, 5, 6,],
        'poly__degree': [1, 2],
        'bag__max_samples': [0.75, 1.0],
        'bag__max_features': [1.0],
        'bag__bootstrap': [True, False],
        'bag__n_estimators': [100, 150, 250],
        'bag__base_estimator':  [LinearRegression()]
    }

    gcv = GridSearchCV(pipe, params, cv=KFold(random_state=5, shuffle=True))

    gcv.fit(X, y)

    return gcv


def xgboost(X, y):
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25)

    scalers = [MaxAbsScaler(), RobustScaler(), MinMaxScaler(), StandardScaler()]

    pipe = Pipeline([
    ('scaler', MaxAbsScaler()),
    ('pca', PCA()),
    ('poly', PolynomialFeatures()),
    ('XGBoost', XGBRegressor(objective='reg:squarederror',
                            learning_rate=0.2,
                                max_depth=20,
                                n_estimators=100,
                                subsample=0.50,
                                colsample_bytree=0.50,
                                alpha=2,
                                reg_lambda=1,
                                gamma = 5,)

        )])

    params = {
    'scaler': scalers, # DV MaxAbsScaler
    'pca__n_components': [2,3,4, 6,], # 6 9 6max
    'poly__degree': [1, 2], # 1 1 2
    'XGBoost__learning_rate': np.arange(.05, 0.25, 0.01), # 0.2 0.07 0.2
    'XGBoost__n_estimators': np.arange(50, 300, 1), # 991
    #'XGBoost__gamma': np.arange(0.1, 1, 10), # 0.1 0.1 0.1
    # 'XGBoost__reg_lambda': np.arange(0.1, 1, 10), # 0.1
    # 'XGBoost__reg_alpha': np.arange(0.1, 1, 10), # 0.1
    #'XGBoost__subsample': np.arange(.05, 1, .05),# 0.35
    #'XGBoost__max_depth': np.arange(3, 20, 1), # 7
    #'XGBoost__colsample_bytree': np.arange(.1, 1.05, .05), #0.95
    }

    gcv = RandomizedSearchCV(pipe,
                            params,
                            n_iter=50,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42)
                    )

    eval_set=[(X_train, y_train), (X_validation, y_validation)]
    
    fit_params = {'XGBoost__early_stopping_rounds':20, 'XGBoost__eval_set':eval_set}

    gcv.fit(X, y, **fit_params)
    return gcv


def xgboost2(X, y):
    X = X.values
    y = y.values

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25)
    
    

    model =  XGBRegressor(objective='reg:squarederror',
                            learning_rate=0.2,
                                max_depth=10,
                                n_estimators=100,
                                subsample=0.50,
                                colsample_bytree=0.50,
                                alpha=2,
                                reg_lambda=1,
                                gamma = 5,)

    pipe = Pipeline([
    #('scaler', RobustScaler()),
    ('model', model
        )])

    eval_set=[(X_train, y_train), (X_validation, y_validation)]
    fit_params = {'XGBoost__early_stopping_rounds':20, 'XGBoost__eval_set':eval_set}


    pipe.fit(X_train, y_train,
            model__eval_set = eval_set,
            model__early_stopping_rounds = 20,
    )
    
    model.fit(X_train, y_train,
            eval_set = eval_set,
            early_stopping_rounds = 100,
    )


    return model

X_train, X_validation, y_train, y_validation = train_test_split(eb_Xtrain, eb_ytrain, test_size=0.25)

#?~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#? Eaglebine
#?~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%% #* Modify TrueTemp
# Use Synthetic temperature instead of static temperature
EB.loc[117, ['True Temp, C']] = EB.loc[117, 'Synthetic Temp, C']
# EB.loc[165, ['TrueTemp', 'True Temp_x, C', 'True Temp, C_y']] = 140


#%% #* Split data

eb_features = ['BHT, C',
                'TD, m_x',
                'SurfaceLatitude_NAD27',
                'SurfaceLongitude_NAD27',
                'Depth subsea, m',
                'Depth subsurface, m',
                'MW at Depth(KB), m',
                'GR',
            ]

EB_train = EB.loc[EB.label == 'train']

eb_Xtrain = EB_train[eb_features]
eb_ytrain = EB_train['True Temp, C']


#%% #* Train Model
eb_model = linear_model(eb_Xtrain, eb_ytrain)
# eb_model = xgboost2(eb_Xtrain, eb_ytrain)


# %%
EB_train['Temp_pred'] = eb_model.predict(eb_Xtrain)


mae, mse, r2 = (mean_absolute_error(EB_train['Temp_pred'] , EB_train['True Temp, C']),
                mean_squared_error(EB_train['Temp_pred'] , EB_train['True Temp, C']),
                r2_score(EB_train['Temp_pred'] , EB_train['True Temp, C']))

print(f'MAE: {mae:.2f} \nMSE: {mse:.2f} \nR^2 score: {r2:.2f} \n  ')
# print(eb_model.best_params_)
plot_comparison(EB_train['True Temp, C'], EB_train['Temp_pred']);

# %%
results = eb_model.evals_result()

plt.figure(figsize=(10,7))
plt.plot(results["validation_0"]["rmse"], label="Training loss")
plt.plot(results["validation_1"]["rmse"], label="Validation loss")
plt.axvline(21, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.legend()
#%% #* Get test results and compare with best prediction
EB_test = EB.loc[ EB.label == 'test']
EB_test.loc[:,'TrueTemp']= eb_model.predict(EB_test[eb_features])

plot_comparison_HG(EB_test, best);
# %%


#?~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#? Duvernay
#?~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%% #* Split data

dv_features = [
 'Depth subsea, m',
 'BHT, C',
 'SurfaceLatitude_NAD27',
 'SurfaceLongitude_NAD27',
 'BottomLatitude_NAD27',
 'BottomLongitude_NAD27',
 'GR',
 'oil total cum, bbl',
 'gas total cum, mcf',
 'water total cum, bbl',
 'gor total average',
 'gas maximum, mcf',
 'oil maximum, bbl',
 'water maximum, bbl',
 'yield total average',
 'DST BHT, C',
 'Maximum Shut-in Pressure, kPa',
 'NPHI_SS',
 'NPHI_LS',
 'DPHI_SS',
 'DPHI_LS',
 'PEFZ',
 'RHOB',
 'CAL'
 ]

# DV_train = DV.loc[DV.label == 'train']
DV_train = DV.loc[(DV.label == 'train') & (DV.source != 'static')]

DV_Xtrain = DV_train[dv_features]
DV_ytrain = DV_train['True Temp, C']


#%% #* Train Model
DV_model = linear_model(DV_Xtrain, DV_ytrain)
# DV_model = xgboost2(DV_Xtrain, DV_ytrain)


# %%
DV_train['Temp_pred'] = DV_model.predict(DV_Xtrain)


mae, mse, r2 = (mean_absolute_error(DV_train['Temp_pred'] , DV_train['True Temp, C']),
                mean_squared_error(DV_train['Temp_pred'] , DV_train['True Temp, C']),
                r2_score(DV_train['Temp_pred'] , DV_train['True Temp, C']))

print(f'MAE: {mae:.2f} \nMSE: {mse:.2f} \nR^2 score: {r2:.2f} \n  ')
# print(DV_model.best_params_)
plot_comparison(DV_train['True Temp, C'], DV_train['Temp_pred']);

#%%
results = DV_model.evals_result()

plt.figure(figsize=(10,7))
plt.plot(results["validation_0"]["rmse"], label="Training loss")
plt.plot(results["validation_1"]["rmse"], label="Validation loss")
plt.axvline(21, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.legend()
# %%
#%% #* Get test results and compare with best prediction
DV_test = DV.loc[ DV.label == 'test']
DV_test.loc[:,'TrueTemp']= DV_model.predict(DV_test[dv_features])

plot_comparison_HG(DV_test, best);

#%%
#?~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#? Submission
#?~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

submission = pd.concat([DV_test[['UWI', 'TrueTemp']], EB_test[['UWI', 'TrueTemp']]], axis=0)
submission.to_csv('prediction/predictions.csv')
# %%
DV[['Surf']]
# %%
