# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

#Split the data into independent variables and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#Categorical variables (State column is a categorical variable)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
#Apply label encoder on the State column
X[:, 3 ] = labelEncoder_X.fit_transform(X[:, 3])

#Add dummy variables to replace the state column
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()


#Avoiding the dummy variable trap, so we remove one of the dummy variables, here we will remove the first one (column 0)
#Some libraries take care of that already but
#some libraries don't, so we have to do it manually
X = X[: , 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Fitting the model into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Now predict on the training set first
y_pred = regressor.predict(X_test)


#Build the optimal model
import statsmodels.formula.api as sm

#We need to add a new column to independent variables matrix to be the constant of the linear regression equation
#Some libraries take care of the constant but statsmodels library don't, so we need to add it

#If we don't add the column, the API will not add the constant part to the equation

#Append the 1s column to X
#X = np.append(X, np.ones( (len(X),1) ).astype(int) ,1) #this will add the new column to the end of the matrix,
# but we need it to be the first column, so we append X to the ONES column

X =  np.append(np.ones((len(X),1)).astype(int),X,1)
#since the default data type of np.ones is float, we need to conver to int
#To specify the column, 0 for row, 1 for column

#Start backward elimination process

#X_opt matrix will contain the independent variables with the most statistical significance for our model
#which mean the variables that have the most effect on the dependent variables

#X_opt first starts with all features we have, and we move some of them in the next steps

X_varaibles = [0,1,2,3,4,5]
X_opt = X[ : , X_varaibles]

#Specify significance level
SL = 0.05

should_continue = True
while should_continue:
    #Fit the model with all predictors in it
    regressor_OLS = sm.OLS(endog = y,exog = X_opt)
    regressor_OLS = regressor_OLS.fit()
    regressor_OLS.summary()

    pvalues = regressor_OLS.pvalues;
    print(pvalues)
    print(type(pvalues))

    #get the maximim pvalue
    max_pvalue = max(pvalues)
    print(max_pvalue)

    max_pvalue_index = -1

    if(max_pvalue > SL):
        # Get the index of the maximum pvalue in the list of indexes
        max_pvalue_index = np.where(pvalues == max_pvalue) #returns an ndarray (multi-dimensional array)
        print(max_pvalue_index[0][0])
        max_pvalue_index = max_pvalue_index[0][0]

    else:
        should_continue = False
        break;

    print("max_pvalue_index ",max_pvalue_index)

    #Now remove that element from the X_varaibles and reinitialize X_opt
    del X_varaibles[max_pvalue_index]
    print('Updated X_varaibles ',X_varaibles)
    X_opt = X[:, X_varaibles]



regressor_OLS.summary()

#We have the optimal model, now predict
y_pred = regressor_OLS.predict(X_opt)

