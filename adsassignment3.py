# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
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

    x_data = np.array(col_list[4:-1])

    df.drop(['Country Name', 'Country Code', 'Indicator Code', 'Unnamed: 67'],
            axis=1, inplace=True)
    transposed_df = df_filtered.transpose()
    transposed_df.columns = transposed_df.iloc[0]
    transposed_df = transposed_df.drop('Indicator Name')
    y_data = np.array(
        transposed_df['Urban population (% of total population)'])
    return x_data, y_data, transposed_df


# Load data for Aruba and Afghanistan
df = pd.read_csv('API_19_DS2_en_csv_v2_5455435.csv', skiprows=4)
# filter data for Aruba and Afghanistan and remove any NaN or inf values

country = 'Germany'
df_filtered = filter(df, country)

x_data, y_data, trans_df = findxandy(df_filtered)

# get data for urban population as a percentage of total population
col_names = df_filtered.columns
col_list = list(col_names)

x_data = np.array(col_list[4:-1])

df_filtered.drop(['Country Name', 'Country Code','Indicator Code', 'Unnamed: 67'], axis=1, inplace=True)
transposed_df = df_filtered.transpose()
transposed_df.columns = transposed_df.iloc[0]
transposed_df =transposed_df.drop('Indicator Name')
y_data = np.array(transposed_df['Urban population (% of total population)'])

# df = df.loc[:, ['Country Name', 'Country Code', 'Indicator Name', '1960', '1970', '1980', '1990', '2000', '2010', '2020']]
# df = df[df['Indicator Name'] == 'Urban population (% of total population)']
# df = df[df['Country Code'].isin(['ABW', 'AFG'])].set_index('Country Code').T.reset_index()
# df.columns.name = None
# df = df.rename(columns={'index': 'Year'})
# Normalize data
df_norm = (df.iloc[:, 1:] - df.iloc[:, 1:].mean()) / df.iloc[:, 1:].std()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(df_norm.iloc[:, 1:])
clusters = kmeans.predict(df_norm.iloc[:, 1:])
df_norm['cluster'] = clusters
# Plot cluster membership and cluster centers
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df_norm.iloc[:, 1], df_norm.iloc[:, 2], c=clusters, cmap='coolwarm')
ax.set_xlabel('Aruba Urban Population (% of total population)')
ax.set_ylabel('Afghanistan Urban Population (% of total population)')
ax.set_title('Urban Population (% of total population) Clusters')
handles, labels = scatter.legend_elements()
legend = ax.legend(handles, ['Cluster 1', 'Cluster 2'], loc='lower left', title='Clusters')
ax.add_artist(legend)

# Plot cluster centers
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, color='black')
for i, c in enumerate(centers):
    ax.annotate(f'Cluster {i+1}', (c[0], c[1]), xytext=(c[0]-0.05, c[1]+0.05), fontsize=12)
    
plt.show()
# fit a logistic function to the data for Aruba and Afghanistan
from scipy.optimize import curve_fit

def logistic(x, A, B, C, k):
    return A / (1 + np.exp(-k*(x-C))) + B

# filter data for Aruba and Afghanistan
df_filtered = df[(df['Country Name'] == 'Aruba') | (df['Country Name'] == 'Afghanistan')]

# get data for urban population as a percentage of total population
x_data = np.array(df_filtered['Year'])
y_data = np.array(df_filtered['Urban population (% of total population)'])

# normalize data
x_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
y_norm = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

# fit the logistic function to the normalized data
popt, pcov = curve_fit(logistic, x_norm, y_norm)

# create a range of x values for the predicted data
x_predict = np.linspace(0, 1, num=100)

# predict y values for the logistic function using the fitted parameters
y_predict = logistic(x_predict, *popt)

# unnormalize the predicted data
y_predict = y_predict * (np.max(y_data) - np.min(y_data)) + np.min(y_data)

# calculate the lower and upper bounds of the confidence interval
err_lower, err_upper = err_ranges(x_norm, y_norm, logistic, popt, pcov, alpha=0.05)

# unnormalize the error bounds
err_lower = err_lower * (np.max(y_data) - np.min(y_data)) + np.min(y_data)
err_upper = err_upper * (np.max(y_data) - np.min(y_data)) + np.min(y_data)

# plot the data and the best fitting logistic function with error bounds
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_data, y_data)
ax.plot(x_predict * (np.max(x_data) - np.min(x_data)) + np.min(x_data), y_predict, color='red')
ax.fill_between(x_predict * (np.max(x_data) - np.min(x_data)) + np.min(x_data), err_lower, err_upper, alpha=0.2, color='gray')
ax.set_xlabel('Year')
ax.set_ylabel('Urban population (% of total population)')
ax.set_title('Logistic Fit to Urban Population Data for Aruba and Afghanistan')
plt.show()
