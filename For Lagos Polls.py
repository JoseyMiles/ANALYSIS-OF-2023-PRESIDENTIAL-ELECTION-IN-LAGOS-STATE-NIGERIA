#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("cleaned_Lagos_geocode.csv")


# In[3]:


df.info()


# In[4]:


def balltree_cluster(df, lat_col = 'Latitude', long_col = 'Longitude', dist_threshold = 1):
    from sklearn.neighbors import BallTree
    #convert the latitude and longitude columns to radians as used by balltree
    df['latitude_rad'] = np.radians(df[lat_col])
    df['longitude_rad'] = np.radians(df[long_col])
    
    coords = df[['latitude_rad', 'longitude_rad']].values

    # Create a BallTree for efficient spatial queries
    tree = BallTree(coords, metric='haversine')
    #calculate the search radius in radians
    radius = dist_threshold / 6371.0 #Earth's radius in km
    # query the tree for each point to find neighbors within the specified distance
    indices = tree.query_radius(coords, r=radius)
    
    #creating clusters for all units
    #create a placeholder cluster for all rows and assign as -1
    cluster = [-1] * len(df)
    #initial cluster_id = 0
    cluster_id = 0
    
    for i, neighbors in enumerate(indices):
        #check if the cluster has been assigned
        if cluster[i] == -1: #if the cluster is yet to be assigned
            #loop through the neigbors and assign clusters to all neighbors of the iteration
            for neighbor in neighbors:
                cluster[neighbor] = cluster_id
            #increment to the next cluster
            cluster_id += 1
            
    return cluster
    
cluster =  balltree_cluster(df, lat_col = 'Latitude', long_col = 'Longitude', dist_threshold = 1)
df['cluster']= cluster


# In[5]:


df.head()


# In[6]:


# Calculate the mean votes for each party within each cluster
# Group by 'Cluster' and calculate the mean for 'APC', 'LP', 'PDP', and 'NNPP'
cluster_means = df.groupby('cluster')[['APC', 'LP', 'PDP', 'NNPP']].mean().reset_index()


# In[7]:


# Rename columns for clarity
cluster_means.columns = ['cluster', 'APC_mean', 'LP_mean', 'PDP_mean', 'NNPP_mean']


# In[8]:


# Merge the cluster means back into the original dataframe
df = df.merge(cluster_means, on='cluster', how='left')


# In[9]:


# Calculate the deviation from the cluster mean for each party
df['APC_deviation'] = df['APC'] - df['APC_mean']
df['LP_deviation'] = df['LP'] - df['LP_mean']
df['PDP_deviation'] = df['PDP'] - df['PDP_mean']
df['NNPP_deviation'] = df['NNPP'] - df['NNPP_mean']


# In[10]:


# Calculate the outlier score for each party (absolute deviation)
df['APC_outlier_score'] = df['APC_deviation'].abs()
df['LP_outlier_score'] = df['LP_deviation'].abs()
df['PDP_outlier_score'] = df['PDP_deviation'].abs()
df['NNPP_outlier_score'] = df['NNPP_deviation'].abs()


# In[11]:


# If you want to export to a csv file, use this line of code below.
#df.to_csv('analyzed_lagos_polls.csv', index=False)


# In[12]:


# Sort the dataset by the outlier scores for each party
sorted_by_APC = df.sort_values(by='APC_outlier_score', ascending=False).reset_index(drop=True)
sorted_by_LP = df.sort_values(by='LP_outlier_score', ascending=False).reset_index(drop=True)
sorted_by_PDP = df.sort_values(by='PDP_outlier_score', ascending=False).reset_index(drop=True)
sorted_by_NNPP = df.sort_values(by='NNPP_outlier_score', ascending=False).reset_index(drop=True)


# In[13]:


# Display the top 3 outliers for each party
sorted_by_APC_top3 = sorted_by_APC[['PU-Code', 'APC', 'APC_mean', 'APC_deviation', 'APC_outlier_score']].head(3)
sorted_by_LP_top3 = sorted_by_LP[['PU-Code', 'LP', 'LP_mean', 'LP_deviation', 'LP_outlier_score']].head(3)
sorted_by_PDP_top3 = sorted_by_PDP[['PU-Code', 'PDP', 'PDP_mean', 'PDP_deviation', 'PDP_outlier_score']].head(3)
sorted_by_NNPP_top3 = sorted_by_NNPP[['PU-Code', 'NNPP', 'NNPP_mean', 'NNPP_deviation', 'NNPP_outlier_score']].head(3)


# In[14]:


# Print the top 3 outliers for each party
print("Top 3 APC Outliers:")
print(sorted_by_APC_top3)
print("\nTop 3 LP Outliers:")
print(sorted_by_LP_top3)
print("\nTop 3 PDP Outliers:")
print(sorted_by_PDP_top3)
print("\nTop 3 NNPP Outliers:")
print(sorted_by_NNPP_top3)


# In[15]:


# Visualization: Distribution of votes for each party
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['APC', 'LP', 'PDP', 'NNPP']])
plt.title('Distribution of Votes for Each Party')
plt.xlabel('Party')
plt.ylabel('Votes')
plt.show()


# In[16]:


# Visualization: Outlier Scores for each party
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

sns.histplot(df['APC_outlier_score'], bins=30, kde=True, ax=axes[0, 0], color='blue')
axes[0, 0].set_title('APC Outlier Scores')
axes[0, 0].set_xlabel('Outlier Score')
axes[0, 0].set_ylabel('Frequency')

sns.histplot(df['LP_outlier_score'], bins=30, kde=True, ax=axes[0, 1], color='green')
axes[0, 1].set_title('LP Outlier Scores')
axes[0, 1].set_xlabel('Outlier Score')
axes[0, 1].set_ylabel('Frequency')

sns.histplot(df['PDP_outlier_score'], bins=30, kde=True, ax=axes[1, 0], color='red')
axes[1, 0].set_title('PDP Outlier Scores')
axes[1, 0].set_xlabel('Outlier Score')
axes[1, 0].set_ylabel('Frequency')

sns.histplot(df['NNPP_outlier_score'], bins=30, kde=True, ax=axes[1, 1], color='purple')
axes[1, 1].set_title('NNPP Outlier Scores')
axes[1, 1].set_xlabel('Outlier Score')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[17]:


# Visualization: Top 5 outliers for APC
plt.figure(figsize=(5, 3))
sns.barplot(x='PU-Code', y='APC_outlier_score', data=sorted_by_APC_top3, palette='Blues_d')
plt.title('Top 3 APC Outliers')
plt.xlabel('PU-Code')
plt.ylabel('APC Outlier Score')
plt.show()


# In[18]:


# Visualization: Top 5 outliers for LP
plt.figure(figsize=(5, 3))
sns.barplot(x='PU-Code', y='LP_outlier_score', data=sorted_by_LP_top3, palette='Greens_d')
plt.title('Top 3 LP Outliers')
plt.xlabel('PU-Code')
plt.ylabel('LP Outlier Score')
plt.show()


# In[19]:


# Visualization: Top 5 outliers for PDP
plt.figure(figsize=(5, 3))
sns.barplot(x='PU-Code', y='PDP_outlier_score', data=sorted_by_PDP_top3, palette='Reds_d')
plt.title('Top 3 PDP Outliers')
plt.xlabel('PU-Code')
plt.ylabel('PDP Outlier Score')
plt.show()


# In[20]:


# Visualization: Top 5 outliers for NNPP
plt.figure(figsize=(5, 3))
sns.barplot(x='PU-Code', y='NNPP_outlier_score', data=sorted_by_NNPP_top3, palette='Purples_d')
plt.title('Top 3 NNPP Outliers')
plt.xlabel('PU-Code')
plt.ylabel('NNPP Outlier Score')
plt.show()


# In[21]:


# Haversine formula to calculate distance between two lat/lon points
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Calculate the deviation of votes from neighboring units (clusters)
def calculate_deviation(df, party, cluster_column='cluster', pu_column='PU-Code'):
    deviation_list = []
    for cluster in df[cluster_column].unique():
        cluster_data = df[df[cluster_column] == cluster]
        cluster_mean = cluster_data[party].mean()
        for index, row in cluster_data.iterrows():
            deviation = abs(row[party] - cluster_mean)
            deviation_list.append((row[pu_column], deviation))
    return pd.DataFrame(deviation_list, columns=[pu_column, f'{party}_Deviation'])

# Calculate deviations for each party
deviations_apc = calculate_deviation(df, 'APC')
deviations_lp = calculate_deviation(df, 'LP')
deviations_pdp = calculate_deviation(df, 'PDP')
deviations_nnpp = calculate_deviation(df, 'NNPP')

# Merge deviations back into the original dataframe
df = df.merge(deviations_apc, on='PU-Code', how='left')
df = df.merge(deviations_lp, on='PU-Code', how='left')
df = df.merge(deviations_pdp, on='PU-Code', how='left')
df = df.merge(deviations_nnpp, on='PU-Code', how='left')

# Check if the columns have been merged correctly
print(df.head())

# Sort by deviations to identify top 3 outliers for each party
sorted_by_APC = df.sort_values(by='APC_Deviation', ascending=False).head(3)
sorted_by_LP = df.sort_values(by='LP_Deviation', ascending=False).head(3)
sorted_by_PDP = df.sort_values(by='PDP_Deviation', ascending=False).head(3)
sorted_by_NNPP = df.sort_values(by='NNPP_Deviation', ascending=False).head(3)

# Function to plot outliers and their clusters for a specific party
def plot_outliers(df, outliers_df, party, title):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='Longitude', y='Latitude', hue='cluster', data=df, palette='viridis', alpha=0.5)
    sns.scatterplot(x='Longitude', y='Latitude', data=outliers_df, s=100, color='red', label='Outliers')
    
    for i, row in outliers_df.iterrows():
        plt.text(row['Longitude'], row['Latitude'], row['PU-Code'], fontsize=9, ha='right', color='red')
    
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

# Add a 'Party' column to distinguish the outliers
sorted_by_APC['Party'] = 'APC'
sorted_by_LP['Party'] = 'LP'
sorted_by_PDP['Party'] = 'PDP'
sorted_by_NNPP['Party'] = 'NNPP'

# Plot the top 3 outliers and their clusters for each party
plot_outliers(df, sorted_by_APC, 'APC', 'APC Top 3 Outliers and Their Clusters')
plot_outliers(df, sorted_by_LP, 'LP', 'LP Top 3 Outliers and Their Clusters')
plot_outliers(df, sorted_by_PDP, 'PDP', 'PDP Top 3 Outliers and Their Clusters')
plot_outliers(df, sorted_by_NNPP, 'NNPP', 'NNPP Top 3 Outliers and Their Clusters')


# In[ ]:




