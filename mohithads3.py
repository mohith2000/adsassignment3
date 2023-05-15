# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:14:29 2023

@author: user
"""

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
    Filters the DataFrame to include data for a specific country.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        country (str): The name of the country to filter the data for.

    Returns:
        pandas.DataFrame: The filtered DataFrame with data for the specified country.
    """
    df_filtered = df[(df['Country Name'] == country)].replace([np.inf,
                                                               -np.inf],
                                                              np.nan).fillna(0)
    return df_filtered


def findxandy(df):
    """
    Extracts the x and y data from the DataFrame and returns them along with other relevant information.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        tuple: A tuple containing the x data, y data, transposed DataFrame, and years.

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
    Computes the linear equation y = mx + c.

    Args:
        x (numpy.ndarray): The input array of x-values.
        y (numpy.ndarray): The input array of y-values.
        m (float): The slope of the line.
        c (float): The y-intercept of the line.

    Returns:
        numpy.ndarray: The computed y-values corresponding to the input x-values.

    """
    y = m*x + c
    return y


# Loading the Data
df = pd.read_csv('C:/Users/user/Desktop/climate.csv', skiprows=4)

# filter data for Germany and remove any NaN or inf values
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

#Plotting the figure
plt.figure(figsize=(10,6),dpi=500)

# Set the plot title and axis labels
plt.title('Energy use (kg of oil equivalent per capita) vs. Urban Population for one Country (Germany)')
plt.xlabel('Energy use (kg of oil equivalent per capita)')
plt.ylabel('Urban population (% of total population)')

# Create a scatter plot for each cluster with a different color and label
#for i in range(kmeanscluster.n_clusters):
    
#plotting the scatter plot 
plt.scatter(trans_df1['Energy use (kg of oil equivalent per capita)'],
            trans_df1['Urban population (% of total population)'],
            c=labels, cmap='tab10')
#Setting Y limit
plt.ylim(30,100)
#Setting X limit
plt.xlim(1100,5000)

# Add a legend to the plot with the cluster labels
plt.legend()

#Saving the Fig
plt.savefig('Total Population vs. Urban Population for all Countries.png', dpi=500)
plt.show()

#######################################################################################

#Plotting the line fit using Years for Electricity production and urban population

plt.plot(trans_df['Electricity production from natural gas sources (% of total)'],
         trans_df['Urban population (% of total population)'], 'o', markersize=4)
#transposing the values into X and Y

x = trans_df['Electricity production from natural gas sources (% of total)']
y = trans_df['Urban population (% of total population)']

# Fit a line to the data
popt, pcov = opt.curve_fit(line, x, y)
z = line(x, *popt)

# Calculate the standard deviation of the parameters
perr = np.sqrt(np.diag(pcov))

# Calculate the confidence intervals (95% confidence level)
confidence = 0.95
alpha = 1 - confidence
z_critical = norm.ppf(1 - alpha / 2)
conf_interval = z_critical * perr

# Define the upper and lower bounds of the confidence intervals
z_upper = z + conf_interval[0]
z_lower = z - conf_interval[0]

# Plot the data points as a line
plt.plot(x, z)

# Set the x-axis and y-axis labels and title
plt.xlabel('Electricity production from natural gas sources (% of total)')
plt.ylabel('Urban population (% of total population)')
plt.title('Line Fit: Urban Population vs. Electricity Production (Germany)')

# Add the parameter values to the plot
params_str = f"m = {popt[0]:.2f}, c = {popt[1]:.2f}"
plt.text(0.84, 0.3, params_str, transform=plt.gca().transAxes, ha='center')

# Add the error range to the plot
err_range_str = f"Error Range: +/- {conf_interval[0]:.2f}"
plt.text(0.84, 0.36, err_range_str, transform=plt.gca().transAxes, ha='center')

# Add legend labels
plt.legend(['year', 'Fit'])
plt.savefig('Urban Population vs. Electricity Production (Germany).png', dpi=500)

# Display the plot
plt.show()
