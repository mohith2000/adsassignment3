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
trans_df_norm = (trans_df.iloc[:, 1:] - trans_df.iloc[:, 1:].mean()) / trans_df.iloc[:, 1:].std()

#######################################################################################


# Get the list of all countries in the data
countriesdata = transposeddata['Country Name'].unique()

# Initialize an empty array to store the concatenated data
concat_data1 = np.empty([0, 2])

# Loop over the countries and concatenate the data for the two indicators and the years of interest
for year in years:
    data1 = transposeddata.loc[(transposeddata['Country Name'] == country) & (transposeddata['Indicator Name'] == indicator1name), datayears2].values[0]
    data2 = transposeddata.loc[(transposeddata['Country Name'] == country) & (transposeddata['Indicator Name'] == indicator2name), datayears2].values[0]
    data_country1 = np.column_stack((data1, data2))
    concat_data1 = np.vstack([concat_data1, data_country1])

# Perform KMeans clustering with k=4
kmeanscluster = KMeans(n_clusters= 4, random_state=0).fit(concat_data1)

# Get the cluster labels
labels = kmeanscluster.labels_

# Set the plot title and axis labels
plt.title('Total Population vs. Urban Population for all Countries (2000)')
plt.xlabel('Population, total')
plt.ylabel('Urban population')

# Create a scatter plot for each cluster with a different color and label
for i in range(kmeanscluster.n_clusters):
    plt.scatter(concat_data1[labels == i, 0], concat_data1[labels == i, 1], label='Cluster {}'.format(i+1))

# Add a legend to the plot with the cluster labels
plt.legend()
plt.savefig('scatterplotads.png', dpi=300)
# Display the plot
plt.show()



#######################################################################################





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
