# Prediction of Student Grades using supervised learning methods

#Problem Formulation

The data set consist of information about the survey related to the online courses. 
The online interaction and participation give the information about their learning experience. In this web-based education system every 
login activity is being recorded. By analyzing the login activities, we can understand the behavior of students. 
The students with larger percentage of course completion and student who are actively participating in courses have higher chance of getting good grades.
Moreover, students with inactive participation and less explored will have high risk of dropout.
We are predicting the grade performance and risk of dropout students. 
Here, we are applying the supervised learning methods for predicting the results. 

# Data-Preprocessing and feature scalling
We followed some of the steps for data processing and make it ready for implementing on the models:
Elimination of null values,Imputing new values into missing rows and columns and, Converting the categorical data to numerical data using One-hot encoding and label encoder method.
Used standard scallar and mim-max scalling method.

# Applied supervised learning algorithms 
• Supervised models- Lasso regression, Random forest, and Neural network on the dataset.
• The best for this data set was random forest and worst was neural network
• Efficiency varied by 30% where random forest showed up 79%. Efficiency was measured by considering
  accuracy and precision.



