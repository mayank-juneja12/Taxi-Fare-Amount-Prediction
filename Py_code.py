#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
import numpy as np
import sys
from pandas_profiling import ProfileReport
from sklearn.impute import KNNImputer
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# !{sys.executable} -m pip install xgboost
import xgboost as xgb 
from xgboost import XGBRegressor


# In[161]:


#######Importing dataset
dataset = pd.read_csv("train_cab.csv")


# In[162]:


dataset.describe()


# In[163]:


dataset.head()


# In[164]:


dataset.dtypes


# In[165]:


dataset.isnull().sum()


# In[166]:


###Typecasting variables######
dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'], errors='coerce')
dataset['fare_amount'] = pd.to_numeric(dataset['fare_amount'], errors='coerce')
dataset.dtypes


# In[167]:


pd.isnull(dataset).sum()


# In[168]:


####Splitting datetime variable#######
dataset['pickup_year'] = dataset['pickup_datetime'].dt.year
dataset['pickup_month'] = dataset['pickup_datetime'].dt.month
dataset['pickup_day'] = dataset['pickup_datetime'].dt.day
dataset['pickup_weekday'] = dataset['pickup_datetime'].dt.dayofweek
dataset['pickup_hour'] = dataset['pickup_datetime'].dt.hour


# In[169]:


#######Dropping datetime variable
dataset.drop(['pickup_datetime'], axis=1, inplace=True)
dataset.head()


# In[64]:


#######Generating EDA pandas profiling report
dataset.profile_report()


# In[170]:


dataset.dropna(inplace=True)
dataset.shape


# In[171]:


######Setting irrelevant values to NaN ########

dataset.loc[dataset['fare_amount'] < 0, ['fare_amount']] = np.nan
dataset.loc[dataset['passenger_count'] > 7, ['passenger_count']] = np.nan
dataset['pickup_latitude'].loc[lambda x: (x < -90) | (x > 90)] = np.nan
dataset['pickup_longitude'].loc[lambda x: (x < -180) | (x > 180)] = np.nan
dataset['dropoff_latitude'].loc[lambda x: (x < -90) | (x > 90)] = np.nan
dataset['dropoff_longitude'].loc[lambda x: (x < -180) | (x > 180)] = np.nan


# In[172]:


dataset.isnull().sum()


# In[173]:


####Converting passenger count to categorical and also removing decimal value by omitting them from categories argument
dataset['passenger_count'] = pd.Categorical(dataset.passenger_count, ordered=True, categories = [0, 1, 2, 3, 4, 5, 6])
dataset['passenger_count'].isnull().sum()


# In[174]:


dataset.to_csv('py_data_before_out.csv', index=False)


# In[175]:


######OUTLIER ANALYSIS##########
#lat_limit : 40.5747 41.0309
#lon_limit: -74.1816 -73.6522
dataset['pickup_latitude'].loc[lambda x: (x < 40.5747) | (x > 41.0309)] = np.nan
dataset['pickup_longitude'].loc[lambda x: (x < -74.1816) | (x > -73.6522)] = np.nan
dataset['dropoff_latitude'].loc[lambda x: (x < 40.5747) | (x > 41.0309)] = np.nan
dataset['dropoff_longitude'].loc[lambda x: (x < -74.1816) | (x > -73.6522)] = np.nan


# In[146]:


dataset.isnull().sum()
dataset.reset_index(drop=True, inplace=True)
dataset.profile_report()


# In[176]:


#####Feature Engineering
####Creating a new variable Distance
def haversine_vectorize(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    m = 6378100 * dist 
    return m


# In[177]:


dis = haversine_vectorize(dataset['pickup_longitude'], dataset['pickup_latitude'], dataset['dropoff_longitude'], dataset['dropoff_latitude'])


# In[178]:


dataset['Distance'] = dis


# In[150]:


#####Dropping co-ordinate columns 
dataset.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1, inplace=True)
dataset.columns


# In[115]:


fa_out = sns.boxplot(y = "fare_amount", data = dataset)
fa_out


# In[116]:


sns.distplot(dataset['Distance'])
dataset['Distance'].describe()


# In[179]:


####Setting outliers to NA
dataset['fare_amount'].loc[lambda x: (x == 0) | (x > 100)] = np.nan
dataset['Distance'].loc[lambda x: (x <= 0)] = np.nan


# In[180]:


dataset.isnull().sum()


# In[181]:


###Dropping NA's from fare_amount and passenger_count
dataset.dropna(subset = ["fare_amount", "passenger_count"], inplace = True)


# In[182]:


#######Converting variables to categorical
dataset['pickup_year'] = pd.Categorical(dataset.pickup_year, ordered= False)
dataset['pickup_month'] = pd.Categorical(dataset.pickup_month, ordered=False)
dataset['pickup_day'] = pd.Categorical(dataset.pickup_day, ordered=False)
dataset['pickup_weekday'] = pd.Categorical(dataset.pickup_weekday, ordered=False)
dataset['pickup_hour'] = pd.Categorical(dataset.pickup_hour, ordered=False)


# In[183]:


dataset.dtypes


# In[184]:


####Generating Pandas Profiling Report
dataset.reset_index(drop=True, inplace=True)
dataset.profile_report()


# In[122]:


####Creating a copy of dataset
dat1 = dataset.copy(deep=True)


# In[123]:


###KNNImputation for NaN
imputer = KNNImputer()
print(dat1.iloc[1246, 7]) ##10591.40
dat1.iloc[1246, 7] = np.nan
dataknn = imputer.fit_transform(dat1)


# In[124]:


print(dataknn[1246, 7]) ###3762.422158173783
type(dataknn)
dat1.dtypes


# In[125]:


########Converting dataknn from ndarray to DataFrame
dataknn = pd.DataFrame(data = dataknn, columns=dataset.columns)


# In[126]:


dataknn.profile_report()


# In[26]:


###Writing data fro Exploratory Data Analysis
dataknn.to_csv("py_data_for_eda.csv", index = False)
dataknn.head()


# In[27]:


####FEATURE ENGINEERING#############
def session(x):
    if((x >= 1) & (x <= 5)): return 'Midnight'
    if((x >= 6) & (x <= 13)): return 'Morning'
    if((x >= 14) & (x <= 17)): return 'Afternoon' 
    if((x >= 18) & (x <= 21)): return 'Evening'
    if((x >= 22) | (x == 0)): return 'Night'
    
###Creating new variable pickup_session
dataknn['pickup_session'] = dataknn['pickup_hour'].apply(session)


# In[28]:


dataknn['pickup_session']


# In[29]:


def season(x):
    if((x >= 2) & (x <= 5)): return 'Spring'
    if((x >= 6) & (x <= 7)): return 'Summer'
    if((x >= 8) & (x <= 10)): return 'Autumn'
    if((x >= 11) | (x == 1)): return 'Winter'

###Creating new variable pickup_session
dataknn['pickup_season'] = dataknn['pickup_month'].apply(season)


# In[30]:


dataknn['pickup_season'] = pd.Categorical(dataknn.pickup_season, ordered=False)
dataknn['pickup_session'] = pd.Categorical(dataknn.pickup_session, ordered = False)


# In[31]:


dataknn['pickup_year'] = pd.Categorical(dataknn.pickup_year, ordered = False)


# In[32]:


###Dropping redundant variables
dataknn.drop(['pickup_hour', 'pickup_month', 'pickup_day', 'pickup_weekday'], inplace=True, axis = 1)


# In[33]:


dataknn['passenger_count'] = pd.Categorical(dataknn.passenger_count, ordered = True)


# In[34]:


categorical_data = dataknn.select_dtypes(include= 'category')
categorical_data.describe()


# In[35]:


##########ANOVA Test############
anova_data = pd.concat([categorical_data, dataknn['fare_amount']], axis = 1)


# In[38]:


p_value = []
aov_data = pd.DataFrame({
    'Column_Names' : anova_data.columns.tolist()
})
for col in categorical_data.columns.tolist():
    mod = ols('fare_amount ~ {}'.format(col), data=anova_data).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
#     print(aov_table)
    p_value.append(aov_table.iloc[0, 3]) 
aov_data = pd.concat([aov_data, pd.Series(p_value)], axis = 1)


# In[39]:





# In[36]:


########Splitting dataknn in numerical and categorical
numerical_data = dataknn.select_dtypes(include= 'float64')
scaler = MinMaxScaler()
numerical_data['Distance'] = scaler.fit_transform(np.asarray(numerical_data['Distance']).reshape(-1, 1))
numerical_data['Distance'].describe()


# In[81]:


####Creating Dummy Variables for Categorical Data
ohe_data = pd.get_dummies(data = categorical_data.iloc[:, :4], drop_first=True)


# In[82]:


####Combining the scaled data and One Hot Encoding data
final_data = pd.concat([numerical_data, ohe_data], axis = 1)


# In[83]:


final_data


# In[84]:


#####Splitting final_data to Training and Validation Data
X_train, X_valid, Y_train, Y_valid = train_test_split(final_data.iloc[:, final_data.columns != 'fare_amount'], 
                                                    final_data['fare_amount'], test_size = 0.2, 
                                                    random_state = 123)


# In[85]:


#####LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, Y_train)


# In[86]:


######Model Evaluation Metrics
def metrics(train_pred, valid_pred, y_train, y_valid):
    
    # Root mean squared error
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
    
    # Calculate absolute percentage error
    train_ape = abs((y_train - train_pred) / y_train)
    valid_ape = abs((y_valid - valid_pred) / y_valid)
    
    # Account for y values of 0
    train_ape[train_ape == np.inf] = 0
    train_ape[train_ape == -np.inf] = 0
    valid_ape[valid_ape == np.inf] = 0
    valid_ape[valid_ape == -np.inf] = 0
    
    train_mape = 100 * np.mean(train_ape)
    valid_mape = 100 * np.mean(valid_ape)
    
    return train_rmse, valid_rmse, train_mape, valid_mape


def evaluate(model, X_train, X_valid, y_train, y_valid):
    
    # Make predictions
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    
    # Get metrics
    train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,
                                                             y_train, y_valid)
    
    print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')
    print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')


# In[87]:


evaluate(lr, X_train, X_valid, Y_train, Y_valid)


# In[88]:


######RANDOM FOREST##########
rf = RandomForestRegressor(random_state=123)
rf.fit(X_train, Y_train)


# In[89]:


evaluate(rf, X_train, X_valid, Y_train, Y_valid)


# In[90]:


#######Parameter Tuning########
random_grid = {'n_estimators': range(100,500,100),
               'max_depth': range(5,20,1),
               'min_samples_leaf':range(2,5,1),
               'max_features':['auto','sqrt','log2'],
               'bootstrap': [True, False],
               'min_samples_split': range(2,5,1)}


# In[105]:


########Randomized Search CV for finding the optimal parameters
rf_cv = RandomizedSearchCV(rf, random_grid, cv=5, scoring ='neg_mean_absolute_error', random_state=123)
rf_cv.fit(final_data.iloc[:, final_data.columns != 'fare_amount'], final_data['fare_amount'])
print("Tuned Random Forest Parameters: {}".format(rf_cv.best_params_))
print("Best score is {}".format(rf_cv.best_score_))


# In[106]:


###Using hyperparameters for training the data
rf = RandomForestRegressor(n_estimators = 200, 
                           min_samples_split = 4, 
                           min_samples_leaf = 4, 
                           max_features = 'auto', 
                           max_depth = 6, 
                           bootstrap = False)
rf.fit(X_train, Y_train)
evaluate(rf, X_train, X_valid, Y_train, Y_valid)


# In[107]:


#########EXTREME GRADIENT BOOSTING(XGBOOST)#########
xg =xgb.XGBRegressor()
# X_train['passenger_count'] = pd.to_numeric(X_train['passenger_count'])
# X_valid['passenger_count'] = pd.to_numeric(X_valid['passenger_count'])
xg.fit(X_train, Y_train)
train_pred = xg.predict(X_train)
valid_pred = xg.predict(X_valid)

train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,
                                                             Y_train, Y_valid)
    
print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')
print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')


# In[111]:


#######Setting the parameters range for tuning############
params = {'objective':['reg:linear'],
          'n_estimators': range(100,3000,100),
          'max_depth': range(3,10,1),
          'subsample': np.arange(0.1,1,0.1),
          'colsample_bytree': np.arange(0.1,1,0.1),
          'colsample_bylevel': np.arange(0.1,1,0.1),
          'colsample_bynode': np.arange(0.1,1,0.1),
          'learning_rate': np.arange(0.05, 0.3, 0.05)}



# In[112]:


#########Using Randomized Search CV to find the optimal hyperparameters

p_search = RandomizedSearchCV(xg, params, cv=5, scoring='neg_root_mean_squared_error', random_state=123)
p_search.fit(final_data.iloc[:, final_data.columns != 'fare_amount'], final_data['fare_amount'])


# In[113]:


print("Tuned XGBoost Parameters: {}".format(p_search.best_params_))
print("Best score is {}".format(p_search.best_score_))


# In[114]:


#######Using the best parameters for training
xg =xgb.XGBRegressor(objective ="reg:linear", colsample_bytree = 0.8, learning_rate = 0.05, 
                     max_depth = 3, n_estimators = 300, colsample_bynode = 0.7000000000000001, 
                     colsample_bylevel=0.5, subsample =  0.7000000000000001)
xg.fit(X_train, Y_train)
train_pred = xg.predict(X_train)
valid_pred = xg.predict(X_valid)

train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,
                                                             Y_train, Y_valid)
    
print(f'Training:   rmse = {round(train_rmse, 2)} \t mape = {round(train_mape, 2)}')
print(f'Validation: rmse = {round(valid_rmse, 2)} \t mape = {round(valid_mape, 2)}')


# In[119]:


######Applying the XGBoost model to test data##############
test = pd.read_csv("test.csv")


# In[120]:


test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], errors='coerce')
test.dtypes


# In[121]:


####Splitting datetime variable#######
test['pickup_year'] = test['pickup_datetime'].dt.year
test['pickup_month'] = test['pickup_datetime'].dt.month
test['pickup_day'] = test['pickup_datetime'].dt.day
test['pickup_weekday'] = test['pickup_datetime'].dt.dayofweek
test['pickup_hour'] = test['pickup_datetime'].dt.hour


# In[122]:


test.drop(['pickup_datetime'], axis=1, inplace=True)
test.head()


# In[123]:


test.loc[test['passenger_count'] > 7, ['passenger_count']] = np.nan
test['pickup_latitude'].loc[lambda x: (x < -90) | (x > 90)] = np.nan
test['pickup_longitude'].loc[lambda x: (x < -180) | (x > 180)] = np.nan
test['dropoff_latitude'].loc[lambda x: (x < -90) | (x > 90)] = np.nan
test['dropoff_longitude'].loc[lambda x: (x < -180) | (x > 180)] = np.nan


# In[125]:


######Converting passenger count to categorical
test['passenger_count'] = pd.Categorical(test.passenger_count, ordered=True, categories = [0, 1, 2, 3, 4, 5, 6])
test.isnull().sum()


# In[126]:


######OUTLIER ANALYSIS##########
#lat_limit : 40.5747 41.0309
#lon_limit: -74.1816 -73.6522
test['pickup_latitude'].loc[lambda x: (x < 40.5747) | (x > 41.0309)] = np.nan
test['pickup_longitude'].loc[lambda x: (x < -74.1816) | (x > -73.6522)] = np.nan
test['dropoff_latitude'].loc[lambda x: (x < 40.5747) | (x > 41.0309)] = np.nan
test['dropoff_longitude'].loc[lambda x: (x < -74.1816) | (x > -73.6522)] = np.nan


# In[127]:


dis = haversine_vectorize(test['pickup_longitude'], test['pickup_latitude'], test['dropoff_longitude'], test['dropoff_latitude'])


# In[128]:


test['Distance'] = dis


# In[129]:


test.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1, inplace=True)
test.columns


# In[131]:


####Setting outliers to NA
test['Distance'].loc[lambda x: (x <= 0)] = np.nan
test.isnull().sum()


# In[133]:


test.dropna(subset = ["Distance"], inplace = True)


# In[134]:


test['pickup_year'] = pd.Categorical(test.pickup_year, ordered= False)
test['pickup_month'] = pd.Categorical(test.pickup_month, ordered=False)
test['pickup_day'] = pd.Categorical(test.pickup_day, ordered=False)
test['pickup_weekday'] = pd.Categorical(test.pickup_weekday, ordered=False)
test['pickup_hour'] = pd.Categorical(test.pickup_hour, ordered=False)


# In[136]:


test['pickup_session'] = test['pickup_hour'].apply(session)


# In[137]:


test['pickup_season'] = test['pickup_month'].apply(season)


# In[138]:


test['pickup_season'] = pd.Categorical(test.pickup_season, ordered=False)
test['pickup_session'] = pd.Categorical(test.pickup_session, ordered = False)


# In[139]:


test['pickup_year'] = pd.Categorical(test.pickup_year, ordered = False)


# In[140]:


###Dropping redundant variables
test.drop(['pickup_hour', 'pickup_month', 'pickup_day', 'pickup_weekday'], inplace=True, axis = 1)


# In[141]:


test['passenger_count'] = pd.Categorical(test.passenger_count, ordered = True)


# In[142]:


categorical_data_test = test.select_dtypes(include= 'category')
categorical_data_test.describe()


# In[143]:


########Splitting dataknn in numerical and categorical
numerical_data_test = test.select_dtypes(include= 'float64')
scaler = MinMaxScaler()
numerical_data_test['Distance'] = scaler.fit_transform(np.asarray(numerical_data_test['Distance']).reshape(-1, 1))
numerical_data_test['Distance'].describe()


# In[144]:


####Creating Dummy Variables for Categorical Data
ohe_data_test = pd.get_dummies(data = categorical_data_test.iloc[:, :4], drop_first=True)


# In[145]:


####Combining the scaled data and One Hot Encoding data
final_data_test = pd.concat([numerical_data_test, ohe_data_test], axis = 1)


# In[146]:


test_pred = xg.predict(X_train)


# In[147]:


test_pred


# In[ ]:




