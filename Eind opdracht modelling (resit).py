#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 09:14:04 2023

@author: quintenachterberg
"""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#------------Task 1A--------------

df = pd.read_csv('https://raw.githubusercontent.com/QuintenA802/FIML/main/1658049webshop%20(1).csv')
df.info()

df_clean = df.dropna() # Delete the missing values' rows (if one variable has a missing observation, the entire row is excluded from regression analysis).
print(df_clean)

df_clean.isnull().sum() # Check if there are no missing values in the net DataFrame


#------------Task 1B--------------

# Create dummy variables for the Device and Find_website variable
device_dummies = pd.get_dummies(df_clean["Device"], prefix="Device")
find_website_dummies = pd.get_dummies(df_clean["Find_website"], prefix="Find_website")


# Add the dummy variables to the dataframe and delete original columns
df_clean = pd.concat([df_clean, device_dummies, find_website_dummies], axis=1)
df_clean.drop(["Device", "Find_website"], axis=1, inplace=True)

# Fit the regression model with the dummy variables
df_clean.info() #to show the columns to put in the model

model1 = sm.ols('Purchase_Amount ~ Time_Spent_on_Website + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Ease_of_purchase + Age + Device_PC + Device_Mobile + Find_website_Search_Engine + Find_website_Social_Media_Advertisement + Find_website_Friends_or_Family + Find_website_Other', data=df_clean).fit()

# Print the summary of the model
print(model1.summary())


# Detecting outliers - Cook's D 

    # Calculating cook's D value:
    CooksD = model1.get_influence().cooks_distance
    
    # Calculating the sample size
    n = len(df_clean)
    
    # Add a variable with outliers to the original dataset
    df_clean['Outlier'] = CooksD [0] > 4/n
    
    # Select only the ouliers
    df_without_outliers = df_clean[df_clean.Outlier == True]
    
    # Remove the outliers:
    df_without_outliers = df_clean[df_clean.Outlier == False]
    
    # remove rows that remained the number -999 in column Time_Spent_on_Website & Number_of_products_browsed
    df_clean_without_outliers = df_without_outliers[(df_without_outliers['Time_Spent_on_Website'] != -999) & (df_without_outliers['Number_of_products_browsed'] != -999)]


#------------Task 1C--------------

# Checking the correlation with all the columns
corr_matrix_all = df_clean_without_outliers.corr()
print(corr_matrix_all)

# Drop the outliers column since we do not need this column for checking the correlation
DFcor = df_clean_without_outliers.copy()
DFcor.drop('Outlier', axis=1, inplace=True)

# Correlation matrix between all the columns of the dataframe
corr_matrix = DFcor.corr()
print(corr_matrix)

# Create heatmap of the correlation matrix for clarification
sns.heatmap(corr_matrix, annot = True)

# Create a more clear heatmap (Run next two lines at once)
fig, ax = plt.subplots(figsize=(10,5)) 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)

#------------Task 1D--------------

# Graph the independent variable to the dependent variable
y_var =  df_clean_without_outliers['Purchase_Amount']
sns.regplot (y = y_var, x = df_clean_without_outliers. Time_Spent_on_Website, lowess = True, line_kws={"color": "red"}) # Linear relationship, in a straight line
sns.regplot (y = y_var, x = df_clean_without_outliers. Number_of_products_browsed, lowess = True, line_kws={"color": "red"}) # Linear relationship, in a straight line
sns.regplot (y = y_var, x = df_clean_without_outliers. Pictures, lowess = True, line_kws={"color": "red"}) # Linear relationship, but there is a small dent in the middle.
sns.regplot (y = y_var, x = df_clean_without_outliers. Shipping_Time, lowess = True, line_kws={"color": "red"}) # Linear relationship, but there is an even smaller dent in the middle.
sns.regplot (y = y_var, x = df_clean_without_outliers. Review_rating, lowess = True, line_kws={"color": "red"}) # Logorithmic relationship
sns.regplot (y = y_var, x = df_clean_without_outliers. Age, lowess = True, line_kws={"color": "red"}) # Polynomial relationship 

# Transform the data to make it linear:
    # For the logarithmic relationship using LOG:
    LogReview = np.log(df_clean_without_outliers.Review_rating)

    # For the Polynomial relationship
    PowAge = pow(df_clean_without_outliers.Age,2)
    
#To check the outcome
# sns.regplot(y = y_var, x = LogReview, lowess = True, line_kws={"color": "red"})
# sns.regplot(y = y_var, x = PowAge, lowess = True, line_kws={"color": "red"})

# Code to present the model
model2 = sm.ols('Purchase_Amount ~ Time_Spent_on_Website + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Ease_of_purchase + Age + Device_PC + Device_Mobile + Find_website_Search_Engine + Find_website_Social_Media_Advertisement + Find_website_Friends_or_Family + Find_website_Other', data=df_clean_without_outliers).fit()

# Print the summary of the model
print(model2.summary())


#------------Task 2A--------------

Table = Stargazer ([model2])
HTML(Table.render_html())


#------------Task 2B--------------

# Standardizing the independent variables

# Make a copy of the previously used data set
df_s = df_clean_without_outliers.copy()
df_s.info()

# Select columns to standardize
cols_to_standardize = ['Time_Spent_on_Website', 'Number_of_products_browsed', 'Pictures', 'Shipping_Time', 'Review_rating', 'Age']

# Create StandardScaler object
scaler = StandardScaler()

# Fit StandardScaler to selected columns
df_s[cols_to_standardize] = scaler.fit_transform(df_s[cols_to_standardize])
df_s.info()

# Show the standardized DataFrame
print(df_s)


# Now write the model given the 
model3 = sm.ols('Purchase_Amount ~ Time_Spent_on_Website + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Ease_of_purchase + Age + Device_PC + Device_Mobile + Find_website_Search_Engine + Find_website_Social_Media_Advertisement + Find_website_Friends_or_Family + Find_website_Other', data=df_s).fit()

# Print the summary of the model
print(model3.summary())


#------------Task 2C--------------

# Create CustomerX, the new customer whose data is not in the dataframe.
CustomerX = {
    'Time_Spent_on_Website': 723,
    'Number_of_products_browsed': 20,
    'Pictures': 3.4,
    'Shipping_Time': 2.6,
    'Review_rating': 4.5,
    'Ease_of_purchase': 4,
    'Age': 35,
    'Find_website_Friends_or_Family': 1,
    'Find_website_Other': 0,
    'Find_website_Search_Engine': 0,
    'Find_website_Social_Media_Advertisement': 0,
    'Device_Mobile': 0,
    'Device_PC': 1
}

CustomerXdf = pd.DataFrame(CustomerX, index=[0])

# create a StandardScaler object
scaler = StandardScaler()

# fit the scaler on the data
scaler.fit(CustomerXdf[['Time_Spent_on_Website', 'Number_of_products_browsed', 'Pictures', 'Shipping_Time', 'Review_rating', 'Age']])

# transform the data using the scaler
CustomerXdf[['Time_Spent_on_Website', 'Number_of_products_browsed', 'Pictures', 'Shipping_Time', 'Review_rating', 'Age']] = scaler.transform(CustomerXdf[['Time_Spent_on_Website', 'Number_of_products_browsed', 'Pictures', 'Shipping_Time', 'Review_rating', 'Age']])

# make a prediction using the transformed data
prediction = model2.predict(CustomerXdf)

# Print the Purchase Amount of CustomerX, round on 2 decimals
print(f'The predicted Purchase Amount of CustomerX is {round(prediction.iloc[0], 2)}')


# The predicted Purchase Amount of CustomerX is 338.08


#------------Task 2D--------------

# Load your dataset into a pandas dataframe
df = pd.read_csv('https://raw.githubusercontent.com/QuintenA802/FIML/main/1658049webshop%20(1).csv')

# Create an instance of the IterativeImputer class
imputer = IterativeImputer()

# Fit the imputer to your data to learn the imputation model
imputer.fit(df)

# Use the imputer to impute the missing values in your dataset
df_imputed = imputer.transform(df)

# Convert the numpy array output of the imputer to a pandas dataframe
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)


    #------------Task 1B (imputed missing values)--------------
    
    # Create dummy variables for the Device and Find_website variable
    device_dummies = pd.get_dummies(df_imputed["Device"], prefix="Device")
    find_website_dummies = pd.get_dummies(df_imputed["Find_website"], prefix="Find_website")
    
    
    # Add the dummy variables to the dataframe and delete original columns
    df_imputed = pd.concat([df_imputed, device_dummies, find_website_dummies], axis=1)
    df_imputed.drop(["Device", "Find_website"], axis=1, inplace=True)
    
    # Fit the regression model with the dummy variables
    df_imputed.info() #to show the columns to put in the model
    
    model4 = sm.ols('Purchase_Amount ~ Time_Spent_on_Website + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Ease_of_purchase + Age + Device_PC + Device_Mobile + Find_website_Search_Engine + Find_website_Social_Media_Advertisement + Find_website_Friends_or_Family + Find_website_Other', data=df_imputed).fit()
    
    # Print the summary of the model
    print(model4.summary())
    
    
    # Detecting outliers - Cook's D 
    
        # Calculating cook's D value:
        CooksD = model4.get_influence().cooks_distance
        
        # Calculating the sample size
        n = len(df_imputed)
        
        # Add a variable with outliers to the original dataset
        df_imputed['Outlier'] = CooksD [0] > 4/n
        
        # Select only the ouliers
        df_without_outliers = df_impcuted[df_imputed.Outlier == True]
        
        # Remove the outliers:
        df_without_outliers = df_imputed[df_imputed.Outlier == False]
        
        # remove rows that remained the number -999 in column Time_Spent_on_Website & Number_of_products_browsed
        df_clean_without_outliers = df_without_outliers[(df_without_outliers['Time_Spent_on_Website'] != -999) & (df_without_outliers['Number_of_products_browsed'] != -999)]
    
    
    #------------Task 1C (imputed missing values)--------------
    
    # Multicollinearity between the independent variables, by means of a correlation matrix
    corr_matrix = df_clean_without_outliers.corr()
    print(corr_matrix)
    
    
    #------------Task 1D (imputed missing values)--------------
    
    # Graph the independent variable to the dependent variable
    y_var =  df_clean_without_outliers['Purchase_Amount']
    sns.regplot (y = y_var, x = df_clean_without_outliers. Time_Spent_on_Website, lowess = True, line_kws={"color": "red"}) # Linear relationship, in a straight line
    sns.regplot (y = y_var, x = df_clean_without_outliers. Number_of_products_browsed, lowess = True, line_kws={"color": "red"}) # Linear relationship, in a straight line
    sns.regplot (y = y_var, x = df_clean_without_outliers. Pictures, lowess = True, line_kws={"color": "red"}) # Logorithmic relationship
    sns.regplot (y = y_var, x = df_clean_without_outliers. Shipping_Time, lowess = True, line_kws={"color": "red"}) # Logorithmic relationship
    sns.regplot (y = y_var, x = df_clean_without_outliers. Review_rating, lowess = True, line_kws={"color": "red"}) # Linear relationship, but with a small dent in the middle  
    sns.regplot (y = y_var, x = df_clean_without_outliers. Ease_of_purchase, lowess = True, line_kws={"color": "red"}) # Logorithmic relationship
    sns.regplot (y = y_var, x = df_clean_without_outliers. Age, lowess = True, line_kws={"color": "red"}) # Polynomial relationship 
    
    # Transform the data to make it linear:
        # For the logarithmic relationship using LOG:
        LogPictures = np.log(df_clean_without_outliers.Pictures)
        LogShipping_Time = np.log(df_clean_without_outliers.Shipping_Time)
        LogEase_of_purchase = np.log(df_clean_without_outliers.Ease_of_purchase)
        
        # For the Polynomial relationship
        PowAge = pow(df_clean_without_outliers.Age,2)
    
    # Code to present the model
    model5 = sm.ols('Purchase_Amount ~ Time_Spent_on_Website + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Ease_of_purchase + Age + Device_PC + Device_Mobile + Find_website_Search_Engine + Find_website_Social_Media_Advertisement + Find_website_Friends_or_Family + Find_website_Other', data=df_clean_without_outliers).fit()
    
    # Print the summary of the model
    print(model5.summary())
    


