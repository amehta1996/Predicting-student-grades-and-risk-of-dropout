#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Data Mining Project using neural network and random forest for Student grade prediction on the canvas network dataset.
# Loading all the libraries for data pre-processing, exploratory data analysis and visualization.

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
np.random.seed(10)

### Loading the dataset which is in the .tab extension file and can be read using the pandas .read_csv() method.
# While converting the file, we specified that columns 11 to 14 are string.
dff = pd.read_csv('D:\\Group 7 data mining project\\CNPC_1401-1509_DI_v1_1_2016-03-01.csv')
df=dff.copy()
print(df.head())

# Dropping the withheld columns using the .drop() method.
# The original documentation illustrates that in order to de-identify, the columns 'gender' and 'final_cc_cname_DI' were withheld. They might be available in the future releases of the dataset.
df=df.drop('gender', axis=1)
print(df)
df=df.drop('final_cc_cname_DI', axis=1)


## Exploring the Dataset for Data types and Null values. This would be a far better to understand the distribution of null values.
df.isna().mean().round(4) * 100


#Approximately 90 percent of the dataset is missing in the important feature vectors and target variables we want to work with. Therefore, although we will not drop any column entirely, we will drop all rows containing missing values in our target feature.
#And then we will divide the dataset into features and target vectors.
#Here we use a seaborn heatmap tp visualize the null values.


import seaborn as sns
colors = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df.isnull(), cmap=sns.color_palette(colors))


df=df.dropna()
X = df.drop('grade', axis=1)
y = df['grade']
print(X,y)




#Preprocessing
#Replacing the ordinal feature values with numerical values without using a library
#Age - ordinal
X=X.replace({'age_DI': '{}'}, np.nan)
X=X.replace(to_replace =["{19-34}","{34-54}","{55 or older}"],value =[1,2,3])
X['age_DI'].value_counts(dropna= False, sort=True)
#primary_reason - nominal
X['primary_reason']=X['primary_reason'].replace('Missing', np.nan)
X['primary_reason'].value_counts(dropna= False, sort=True)
#learner type - ordinal
X['learner_type']=X['learner_type'].replace('Missing', np.nan)
X=X.replace(to_replace =["Active participant","Active","Passive participant","Passive","Drop-in","Observer"],value =[1,2,3,4,5,6])
X['learner_type'].value_counts(dropna= False, sort=True)
#expected hours week - ordinal
X['expected_hours_week']=X['expected_hours_week'].replace('Missing', np.nan)
X=X.replace(to_replace =["Less than 1 hour","Between 1 and 2 hours","Between 2 and 4 hours","Between 4 and 6 hours","Between 6 and 8 hours","More than 8 hours per week"],value =[1,2,3,4,5,6])
X['expected_hours_week'].value_counts(dropna= False, sort=True)
#Level of education-Ordinal
X['LoE_DI']=X['LoE_DI'].replace('Missing', np.nan)
X=X.replace(to_replace =["None of these","High School or College Preparatory School","Completed 2-year college degree","Some college, but have not finished a degree","Completed 4-year college degree","Some graduate school","Master's Degree (or equivalent)","Ph.D., J.D., or M.D. (or equivalent)"],value =[1,2,3,4,5,6,7,8])
X['LoE_DI'].value_counts(dropna= False, sort=True)









#We assumed the data to be MAR(Missing at random), that is they dont have any relation to other columns.
X=X.replace(to_replace =["2014 Q1","2014 Q2","2014 Q3","2014 Q4",
                        "2015 Q1","2015 Q2","2015 Q3","2015 Q4",
                        "2016 Q1","2016 Q2","2016 Q3"],value =[1,2,3,4,5,6,7,8,9,10,11])
X['start_time_DI'].value_counts(dropna= False, sort=True)
X['course_start'].value_counts(dropna= False, sort=True)
X['course_end'].value_counts(dropna= False, sort=True)
X['last_event_DI'].value_counts(dropna= False, sort=True)
X['nevents'].value_counts(dropna= False, sort=True)
X['ndays_act'].value_counts(dropna= False, sort=True)
X['ncontent'].value_counts(dropna= False, sort=True)
X['nforum_posts'].value_counts(dropna= False, sort=True)
X['course_length'].value_counts(dropna= False, sort=True)
#Filling final missing values
X=X.fillna(0)




#We further realized that discipline and primary reason were not important feature, so we dropped them.
XX=X.copy()
XX=XX.drop(['discipline','primary_reason'],axis=1)
#XX=pd.concat([XX, X_cat], axis=1)
XX






#We then standardised our final dataset with a Standardscaler function and MinMax scaler function to use them in the models to see if we get a better accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
XXMM=MM.fit_transform(XX)
XXMM=pd.DataFrame(np.round(XXMM),columns = XX.columns)

SS=StandardScaler()
XXSS=SS.fit_transform(XX)
XXSS=pd.DataFrame(np.round(XXSS),columns = XX.columns)




#Plotting the Correlation matrix to understand feature importance
plt.figure()
CX=XX.copy()
CX = XX.drop('course_reqs', axis=1)
CX = CX.drop('grade_reqs', axis=1)
CX = CX.drop('registered', axis=1)
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.figsize'] = 80, 80
sns.heatmap(CX.corr(), annot=True)
plt.show()

#Our first model is a RandomForestregresor. To pass the optimum parameters we ran a GridSearchCV algorithm and tested some parameters.
#y=np.ravel(y)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(XX, y, test_size = 0.30, random_state = 10)








gridCVsetup=GridSearchCV(estimator=RandomForestRegressor(),param_grid={'max_depth': range(3,7),'n_estimators': (10, 50, 100, 1000)},cv=5,scoring='neg_mean_squared_error', verbose=True, n_jobs=-1)
gridCV=gridCVsetup.fit(X_Train,Y_Train)
gridBest=gridCV.best_params_
gridBest

#Random Forest with 10-Fold CV
rmfrSetup = RandomForestRegressor(n_estimators=gridBest["n_estimators"], max_depth=gridBest["max_depth"], random_state=10, verbose=True, n_jobs=-1)
rmfrCVscore = cross_val_score(rmfrSetup, X_Train, Y_Train, cv=10, scoring='neg_mean_absolute_error')
rmfr=rmfrSetup.fit(X_Train,Y_Train)
rmffe=rmfr.feature_importances_
rm_y_pred = rmfr.predict(X_Test)
rmfr.score(X_Test,Y_Test)


#Plotting the Metrics and the Negative mean absolute error matrix.
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_Test, rm_y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_Test, rm_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_Test, rm_y_pred)))
print('R Squared(Accuracy)', metrics.r2_score(Y_Test, rm_y_pred))
rmfrCVscore





#Bar plot illustrating features selected by Random Forest 
from matplotlib.pyplot import figure
figure(num=None, figsize=(20, 22), dpi=80, facecolor='w', edgecolor='k')
RandomForestFeatureimportance = pd.Series(rmffe, index= XX.columns)
RandomForestFeatureimportance.nlargest(7).plot(kind='barh')


#Our second model is Lasso CV. 
#Just like the Random Forest, we run an Elastic net first to find optimum values for alpha and then pass to it to the LassoCV model.
X_Train, X_Test, Y_Train, Y_Test = train_test_split(XX, y, test_size = 0.30, random_state = 10)

from sklearn.linear_model import LassoCV,ElasticNetCV

en_cv = ElasticNetCV(cv=5, random_state=10)
en_cv.fit(X_Train, Y_Train)

print('ElasticNetCV alpha:', en_cv.alpha_, 'ElasticNetCV l1_ratio:', en_cv.l1_ratio_)
en_alpha, en_l1ratio = en_cv.alpha_, en_cv.l1_ratio_

en_new_l1ratios = [en_l1ratio * mult for mult in [.9, .95, 1, 1.05, 1.1]]
en_cv = ElasticNetCV(cv=5, random_state=10, l1_ratio=en_new_l1ratios)
en_cv.fit(X_Train, Y_Train)
print('ElasticNetCV alpha:', en_cv.alpha_, 'ElasticNetCV l1_ratio:', en_cv.l1_ratio_)


NewLassoCV=LassoCV(cv=5, random_state=10, alphas=en_new_l1ratios)
NewLassoCV.fit(X_Train,Y_Train)
Nlassocv_pred=NewLassoCV.predict(X_Test)

#Printing the metrics
print("LassoCV Best Alpha Scored: ", NewLassoCV.alpha_)
print("LassoCV Model Accuracy: ", NewLassoCV.score(X_Test, Y_Test))
model_coef1 = pd.Series(NewLassoCV.coef_, index = list(X.columns[:21]))
print("Variables Eliminated: ", str(sum(model_coef1 == 0)))
print("Variables Kept: ", str(sum(model_coef1 != 0)))

#Bar lot illustrating features selected by LassoCV.
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

top_coef = model_coef1.sort_values()
top_coef[top_coef != 0].plot(kind = "barh")
plt.title("Important Features Identified using Lasso")

#A comparison plot of Random Forest and LassoCV
plt.figure()
plt.scatter(rm_y_pred, Y_Test, label='RandomForestRegressor')
plt.scatter(Nlassocv_pred, Y_Test, label='LassoRegressor')
plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Comparison of individual predictions')
plt.show()




#Our third model is a neural network with 6 dense layers, 100 units and relu activation. We tried 'selu and elu' activations but none produced good results.
#We tried different optimizers but settled with 'adam'.
from sklearn import preprocessing
normX=preprocessing.normalize(XX)
X_Train1 = X_Train.drop('course_id_DI', axis=1)
X_Train2 = X_Train1.drop('userid_DI', axis=1)
X_Train3 = X_Train2.drop('registered', axis=1)
X_Train4 = X_Train3.drop('course_reqs', axis=1)
X_Train5 = X_Train4.drop('grade_reqs', axis=1)
X_Train5.shape
X_Test1 = X_Test.drop('course_id_DI', axis=1)
X_Test2 = X_Test1.drop('userid_DI', axis=1)
X_Test3 = X_Test2.drop('registered', axis=1)
X_Test4 = X_Test3.drop('course_reqs', axis=1)
X_Test5 = X_Test4.drop('grade_reqs', axis=1)



y[y<=0.5]=0
y[y>0.5]=1
X_Train, X_Test, Y_Train, Y_Test = train_test_split(XX, y, test_size = 0.30, random_state = 10)


from keras.models import Sequential
from keras.layers import Dense
def neural():
    model = Sequential()
    model.add(Dense(18, activation='sigmoid', input_shape=(X_Train3.shape[1],)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = neural()
model.fit(X_Train3, Y_Train, validation_data=(X_Test3, Y_Test), epochs=200, verbose=2)


scores = model.evaluate(X_Test3, Y_Test)
print(f'Accuracy: {scores[1]} \n Error: {1 - scores[1]}')

