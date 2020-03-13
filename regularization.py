from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

path = '~/Desktop/202/1/'
filename = 'marathon_results_2016.csv'
file_1='marathon_results_2015.csv'
file_2='marathon_results_2017.csv'
df = pd.read_csv(path + filename)
df_1=pd.read_csv(path+file_1)
df_2=pd.read_csv(path+file_2)
df=pd.concat([df,df_1,df_2],axis=0,sort=False)
df.head()

df = df.drop(index=8, columns=['Bib', 'Name', 'City', 'State', 'Country', 'Citizen'], axis=1)
df = df.drop(df.columns[2], axis=1)
df.head()

def time_to_min(string):
    if string is not '-':
        time_segments = string.split(':')
        return int(time_segments[0])*60 + int(time_segments[1]) + np.true_divide(int(time_segments[2]), 60)
    else:
        return -1

def gender_to_numeric(value):
    if value == 'M':
        return 0
    else:
        return 1

df['M/F'] = df['M/F'].apply(lambda x: gender_to_numeric(x))

# remove damaged data
df = df[(~(df['5K'] == '-')) &(~(df['10K'] == '-'))&(~(df['15K'] == '-'))&(~(df['20K'] == '-'))&(~(df['25K'] == '-')) &(~(df['30K'] == '-')) &(~(df['35K'] == '-')) &(~(df['40K'] == '-'))&(~(df['Half'] == '-'))]

# unify data
df['Half'] = df.Half.apply(lambda x: time_to_min(x))
df['Full'] = df['Official Time'].apply(lambda x: time_to_min(x))
df['split_ratio'] = (df['Full'] - df['Half'])/(df['Half'])

df['5K_mins']  = df['5K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = df['10K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = (df['10K_mins'] - df['5K_mins'])

df['15K_mins'] = df['15K'].apply(lambda x: time_to_min(x))
df['15K_mins'] = (df['15K_mins'] - df['10K_mins'] -  df['5K_mins'])

df['20K_mins'] = df['20K'].apply(lambda x: time_to_min(x))
df['20K_mins'] = (df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins'])

df['25K_mins'] = df['25K'].apply(lambda x: time_to_min(x))
df['25K_mins'] = (df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins']
                  - df['5K_mins'])

df['30K_mins'] = df['30K'].apply(lambda x: time_to_min(x))
df['30K_mins'] = (df['30K_mins'] -df['25K_mins'] - df['20K_mins'] - df['15K_mins']
                  - df['10K_mins'] - df['5K_mins'])

df['35K_mins'] = df['35K'].apply(lambda x: time_to_min(x))
df['35K_mins'] = (df['35K_mins'] -df['30K_mins'] - df['25K_mins'] - df['20K_mins']
                    - df['15K_mins'] - df['10K_mins'] - df['5K_mins'])

df['40K_mins'] = df['40K'].apply(lambda x: time_to_min(x))
df['40K_mins'] = (df['40K_mins'] -  df['35K_mins'] - df['30K_mins'] -df['25K_mins']
                    - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] - df['5K_mins'])


columns = ['40K_mins', '35K_mins', '30K_mins', '25K_mins', '20K_mins','15K_mins','10K_mins','5K_mins']
df['avgOf5K'] = df[columns].mean(axis = 1)
df['stdev'] = df[columns].std(axis = 1)

prediction_df = df[[ 'Age', 'M/F', 'Half', 'split_ratio',
                    '40K_mins', '35K_mins', '30K_mins', '25K_mins', '5K_mins','10K_mins','15K_mins','20K_mins',
                    'avgOf5K', 'stdev', 'Full']]
prediction_df.head()


a=1
traindf, testdf = train_test_split(prediction_df, test_size = 0.2)

X1_train = traindf[['Age', 'M/F', '5K_mins']]
y1_train = traindf['10K_mins']
X1_test = testdf[['Age', 'M/F', '5K_mins']]
y1_test = testdf['10K_mins']
X1 = prediction_df[['Age', 'M/F', '5K_mins']].values
y1 = prediction_df['10K_mins'].values
sc1 = StandardScaler()
X1_std = sc1.fit_transform(X1)
poly1 = PolynomialFeatures(degree=3)
X1_poly = poly1.fit_transform(X1_std)
lr_rg = Ridge(alpha=a)
lr_rg.fit(X1_train, y1_train)

y1_train_pred = lr_rg.predict(X1_train)
y1_test_pred = lr_rg.predict(X1_test)

print('\n[Alpha1 = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
            mean_squared_error(y1_train, y1_train_pred),
            mean_squared_error(y1_test, y1_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y1_train, y1_train_pred),
r2_score(y1_test, y1_test_pred)))
                

X2_train = traindf[['Age', 'M/F', '5K_mins','10K_mins']]
y2_train = traindf['15K_mins']
X2_test = testdf[['Age', 'M/F', '5K_mins','10K_mins']]
y2_test = testdf['15K_mins']
X2 = prediction_df[['Age', 'M/F', '5K_mins','10K_mins']].values
y2 = prediction_df['15K_mins'].values
sc2 = StandardScaler()
X2_std = sc2.fit_transform(X2)
poly2 = PolynomialFeatures(degree=3)
X2_poly = poly2.fit_transform(X2_std)
lr_rg = Ridge(alpha=a)
lr_rg.fit(X2_train, y2_train)

y2_train_pred = lr_rg.predict(X2_train)
y2_test_pred = lr_rg.predict(X2_test)

print('\n[Alpha2 = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
            mean_squared_error(y2_train, y2_train_pred),
            mean_squared_error(y2_test, y2_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y2_train, y2_train_pred),
r2_score(y2_test, y2_test_pred)))
 
 
X3_train = traindf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins']]
y3_train = traindf['20K_mins']
X3_test = testdf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins']]
y3_test = testdf['20K_mins']
X3 = prediction_df[['Age', 'M/F', '5K_mins','10K_mins','15K_mins']].values
y3 = prediction_df['20K_mins'].values
sc3 = StandardScaler()
X3_std = sc3.fit_transform(X3)
poly3 = PolynomialFeatures(degree=3)
X3_poly = poly3.fit_transform(X3_std)
lr_rg = Ridge(alpha=a)
lr_rg.fit(X3_train, y3_train)

y3_train_pred = lr_rg.predict(X3_train)
y3_test_pred = lr_rg.predict(X3_test)

print('\n[Alpha3 = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
            mean_squared_error(y3_train, y3_train_pred),
            mean_squared_error(y3_test, y3_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y3_train, y3_train_pred),
r2_score(y3_test, y3_test_pred)))
                
                
X4_train = traindf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins']]
y4_train = traindf['25K_mins']
X4_test = testdf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins']]
y4_test = testdf['25K_mins']
X4 = prediction_df[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins']].values
y4 = prediction_df['25K_mins'].values
sc4 = StandardScaler()
X4_std = sc4.fit_transform(X4)
poly4 = PolynomialFeatures(degree=3)
X4_poly = poly4.fit_transform(X4_std)
lr_rg = Ridge(alpha=a)
lr_rg.fit(X4_train, y4_train)

y4_train_pred = lr_rg.predict(X4_train)
y4_test_pred = lr_rg.predict(X4_test)

print('\n[Alpha4 = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
            mean_squared_error(y4_train, y4_train_pred),
            mean_squared_error(y4_test, y4_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y4_train, y4_train_pred),
r2_score(y4_test, y4_test_pred)))
                
                
X5_train = traindf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins']]
y5_train = traindf['30K_mins']
X5_test = testdf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins']]
y5_test = testdf['30K_mins']
X5 = prediction_df[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins']].values
y5 = prediction_df['30K_mins'].values
sc5 = StandardScaler()
X5_std = sc5.fit_transform(X5)
poly5 = PolynomialFeatures(degree=3)
X5_poly = poly5.fit_transform(X5_std)
lr_rg = Ridge(alpha=a)
lr_rg.fit(X5_train, y5_train)

y5_train_pred = lr_rg.predict(X5_train)
y5_test_pred = lr_rg.predict(X5_test)

print('\n[Alpha5 = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
            mean_squared_error(y5_train, y5_train_pred),
            mean_squared_error(y5_test, y5_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y5_train, y5_train_pred),
r2_score(y5_test, y5_test_pred)))
                    
                
                
X6_train = traindf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins','30K_mins']]
y6_train = traindf['35K_mins']
X6_test = testdf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins','30K_mins']]
y6_test = testdf['35K_mins']
X6 = prediction_df[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins','30K_mins']].values
y6 = prediction_df['35K_mins'].values
sc6 = StandardScaler()
X6_std = sc6.fit_transform(X6)
poly6 = PolynomialFeatures(degree=3)
X6_poly = poly6.fit_transform(X6_std)
lr_rg = Ridge(alpha=a)
lr_rg.fit(X6_train, y6_train)

y6_train_pred = lr_rg.predict(X6_train)
y6_test_pred = lr_rg.predict(X6_test)

print('\n[Alpha6 = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
            mean_squared_error(y6_train, y6_train_pred),
            mean_squared_error(y6_test, y6_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y6_train, y6_train_pred),
r2_score(y6_test, y6_test_pred)))
                
                
X7_train = traindf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins','30K_mins']]
y7_train = traindf['35K_mins']
X7_test = testdf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins','30K_mins']]
y7_test = testdf['35K_mins']
X7 = prediction_df[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins','25K_mins','30K_mins']].values
y7 = prediction_df['35K_mins'].values
sc7 = StandardScaler()
X7_std = sc7.fit_transform(X7)
poly7 = PolynomialFeatures(degree=3)
X7_poly = poly7.fit_transform(X7_std)
lr_rg = Ridge(alpha=a)
lr_rg.fit(X7_train, y7_train)

y7_train_pred = lr_rg.predict(X7_train)
y7_test_pred = lr_rg.predict(X7_test)

print('\n[Alpha7 = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
            mean_squared_error(y7_train, y7_train_pred),
            mean_squared_error(y7_test, y7_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y7_train, y7_train_pred),
r2_score(y7_test, y7_test_pred)))




X8_train = traindf[['Age', 'Half', '5K_mins', '10K_mins', '15K_mins', '20K_mins',
                   '25K_mins', '30K_mins', '35K_mins', '40K_mins','stdev']]
y8_train = traindf['Full']
X8_test = testdf[['Age', 'Half', '5K_mins', '10K_mins', '15K_mins', '20K_mins',
'25K_mins', '30K_mins', '35K_mins', '40K_mins','stdev']]
y8_test = testdf['Full']

X8 = prediction_df[['Age', 'Half', '5K_mins', '10K_mins', '15K_mins', '20K_mins',
'25K_mins', '30K_mins', '35K_mins', '40K_mins','stdev']].values
y8 = prediction_df['Full'].values
sc8 = StandardScaler()
X8_std = sc8.fit_transform(X8)
poly8 = PolynomialFeatures(degree=3)
X8_poly = poly8.fit_transform(X8_std)


lr_rg = Ridge(alpha=a)
lr_rg.fit(X8_train, y8_train)

y8_train_pred = lr_rg.predict(X8_train)
y8_test_pred = lr_rg.predict(X8_test)
print('\n[Alpha8 = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
                mean_squared_error(y8_train, y8_train_pred),
                mean_squared_error(y8_test, y8_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y8_train, y8_train_pred),
r2_score(y8_test, y8_test_pred)))


X_train = traindf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins', 'stdev']]
y_train = traindf['Full']
X_test = testdf[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins', 'stdev']]
y_test = testdf['Full']
X = prediction_df[['Age', 'M/F', '5K_mins','10K_mins','15K_mins','20K_mins', 'stdev']].values
y = prediction_df['Full'].values
sc = StandardScaler()
X_std = sc.fit_transform(X)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_std)
lr_rg = Ridge(alpha=a)
lr_rg.fit(X_train, y_train)

y_train_pred = lr_rg.predict(X_train)
y_test_pred = lr_rg.predict(X_test)

print('\n[Alpha = %d]' % a )
print('MSE train: %.2f, test: %.2f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.2f, test: %.2f' % (
r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))



