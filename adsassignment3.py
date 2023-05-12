# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import KMeans
from scipy.stats import norm


def filter(df, country):
    """
    """
    df_filtered = df[(df['Country Name'] == country)].replace([np.inf,
                                                               -np.inf],
                                                              np.nan).fillna(0)
    return df_filtered


def findxandy(df):
    """
    """
    col_names = df.columns
    col_list = list(col_names)

    years = np.array(col_list[4:-1])

    df.drop(['Country Name', 'Country Code', 'Indicator Code', 'Unnamed: 67'],
            axis=1, inplace=True)
    transposed_df = df_filtered.transpose()
    transposed_df.columns = transposed_df.iloc[0]
    transposed_df = transposed_df.drop('Indicator Name')
    x_data = np.array(
        transposed_df['Agricultural land (sq. km)'])
    y_data = np.array(
        transposed_df['Urban population (% of total population)'])
    
    return x_data, y_data, transposed_df, years


def line(x, y, m, c):
    """
    """
    y = m*x + c
    return y


# Load data for Aruba and Afghanistan
df = pd.read_csv('API_19_DS2_en_csv_v2_5455435.csv', skiprows=4)
# filter data for Aruba and Afghanistan and remove any NaN or inf values

country = 'Germany'
df_filtered = filter(df, country)

x_data, y_data, trans_df, year = findxandy(df_filtered)



# df = df.loc[:, ['Country Name', 'Country Code', 'Indicator Name', '1960', '1970', '1980', '1990', '2000', '2010', '2020']]
# df = df[df['Indicator Name'] == 'Urban population (% of total population)']
# df = df[df['Country Code'].isin(['ABW', 'AFG'])].set_index('Country Code').T.reset_index()
# df.columns.name = None
# df = df.rename(columns={'index': 'Year'})
# Normalize data
#######################################################################################

trans_df.reset_index(inplace=True)
trans_df.rename(columns={'index': 'Year'}, inplace=True)
trans_df1 = trans_df[['Year', 'Energy use (kg of oil equivalent per capita)',
                      'Urban population (% of total population)']]

# Perform KMeans clustering with k=4
kmeanscluster = KMeans(n_clusters= 4, random_state=0).fit(trans_df1)

# Get the cluster labels
labels = kmeanscluster.labels_

plt.figure(figsize=(10,6),dpi=500)
# Set the plot title and axis labels
plt.title('Total Population vs. Urban Population for all Countries (2000)')
plt.xlabel('Population, total')
plt.ylabel('Urban population')

# Create a scatter plot for each cluster with a different color and label
#for i in range(kmeanscluster.n_clusters):

plt.scatter(trans_df1['Energy use (kg of oil equivalent per capita)'],
            trans_df1['Urban population (% of total population)'],
            c=labels, cmap='tab10')
plt.ylim(30,100)
plt.xlim(1100,5000)
# Add a legend to the plot with the cluster labels
plt.legend()
plt.show()



#######################################################################################



plt.scatter(trans_df['Electricity production from natural gas sources (% of total)'],
            trans_df['Urban population (% of total population)'],
               )

x = trans_df['Electricity production from natural gas sources (% of total)']
y = trans_df['Urban population (% of total population)']
# fit a logistic function to the data for Aruba and Afghanistan




# # filter data for Aruba and Afghanistan
# df_filtered = df[(df['Country Name'] == 'Aruba') | (df['Country Name'] == 'Afghanistan')]

# # get data for urban population as a percentage of total population
# x_data = np.array(df_filtered['Year'])
# y_data = np.array(df_filtered['Urban population (% of total population)'])

# # normalize data
# x_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
# y_norm = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

# fit the logistic function to the normalized data
popt, pcov = opt.curve_fit(line, x, y)
sigma = np.sqrt(np.diag(pcov))
z = line(x, *popt)

# # create a range of x values for the predicted data
# x_predict = np.linspace(0, 1, num=100)

# # predict y values for the logistic function using the fitted parameters
# y_predict = logistic(x_predict, *popt)

# # unnormalize the predicted data
# y_predict = y_predict * (np.max(y_data) - np.min(y_data)) + np.min(y_data)

# # calculate the lower and upper bounds of the confidence interval
# err_lower, err_upper = err_ranges(x_norm, y_norm, logistic, popt, pcov, alpha=0.05)

# # unnormalize the error bounds
# err_lower = err_lower * (np.max(y_data) - np.min(y_data)) + np.min(y_data)
# err_upper = err_upper * (np.max(y_data) - np.min(y_data)) + np.min(y_data)

# plot the data and the best fitting logistic function with error bounds
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(x_data, y_data)
# ax.plot(x_predict * (np.max(x_data) - np.min(x_data)) + np.min(x_data), y_predict, color='red')
# ax.fill_between(x_predict * (np.max(x_data) - np.min(x_data)) + np.min(x_data), err_lower, err_upper, alpha=0.2, color='gray')
plt.plot(x, y, "d", markersize=4)
plt.plot(x,z)
ax.set_xlabel('Year')
ax.set_ylabel('Urban population (% of total population)')
ax.set_title('Logistic Fit to Urban Population Data for Aruba and Afghanistan')
plt.show()
