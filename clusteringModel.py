#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:12:13 2020

@author: christopher
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
import os
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import seaborn as sns
sns.set()

# Read dataset

df = pd.read_csv('AutoInsurance.csv')

# Data Cleaning

# Change to datetime format for later
df['Effective To Date'] = pd.to_datetime(df['Effective To Date'], infer_datetime_format=True)
# Set index as customer, and drop columns
df[['Customer', 'Education', 'Total Claim Amount', 'Income', 'Coverage', 'EmploymentStatus', 'Monthly Premium Auto', 'Number of Policies', 'Policy Type']].set_index('Customer')
# Look at numberical columns and distributions
df[['Total Claim Amount','Customer Lifetime Value' ,'Income', 'Number of Open Complaints', 'Monthly Premium Auto']].describe()

# Clean number of complaints column
df['Number of Open Complaints'] = df['Number of Open Complaints'].map({0:0,1:1})
df['Number of Open Complaints'].fillna(1,inplace=True)
df['Number of Open Complaints'] = df['Number of Open Complaints'].astype(int)

#Drop
df.drop(columns=['Effective To Date'],axis=1, inplace=True)

# winsorize customer lifetime value
def winsorize_func(df):
    df = winsorize(df, limits= 0.1, inplace=True)
    return df
df['Customer Lifetime Value'] = winsorize(df['Customer Lifetime Value'], limits= 0.1, inplace=True)


# Income distribution

def distribution_fig(df):
    fig_list = ['Income', 'Customer Lifetime Value']
    for i in fig_list:       
        sns.distplot(df.i)
        plt.title(f'Distribution of {i}')
        plt.xlabel(f'{i}')
        plt.savefig(f'{i} Distribution', dpi=600)
    return df

# Dummies
df= pd.get_dummies(df, columns=['Response', 'State', 'Coverage', 'Education', 
                            'EmploymentStatus',
                            'Gender', 'Location Code', 'Marital Status', 
                            'Policy Type',
                            'Policy', 'Sales Channel', 'Vehicle Class', 
                            'Vehicle Size','Renew Offer Type', 
                            'Number of Open Complaints'], 
                            drop_first=True)

df = df.set_index('Customer')

# First model - no scaled data
X = df # Set X as my Df with dummies and customer as index to run models

model = KMeans(3)
model.fit(X)
y_pred = model.predict(X)
print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
print('Silhouette Score ', silhouette_score(X,y_pred))

centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], 
                    c=['red','blue', 'green'], s=200, alpha=0.5)
plt.title('K Means Cluster Epicenters')
plt.savefig('Unscaled K Means Cluster Epicenters', dpi=600)

# Visualize clusters

plt.scatter(df['Total Claim Amount'], df['Monthly Premium Auto'], c=model.labels_)
plt.title('Total Claim Amount vs Monthly Premium')
plt.xlabel('Total Claim Amount')
plt.ylabel('Monthly Premium Auto')
plt.savefig('Monthly Premium vs Total Claim', dpi=600)

# Scale Data

#Make list of cols with numerical values and fit transform
norm_cols = ['Customer Lifetime Value', 'Income',
             'Monthly Premium Auto', 
             'Months Since Policy Inception',
             'Number of Policies', 'Months Since Last Claim']
scaler = StandardScaler()
for i in norm_cols:
    df[i] = scaler.fit_transform(df[i].values.reshape(-1, 1))

## --- K Means Cluster Model --- ##    
    
X = df
model = KMeans(n_clusters=3)
model.fit(X)
y_pred = model.predict(X)
print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
print('Silhouette Score ', silhouette_score(X,y_pred))
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c=['red','blue', 'green'], s=200, alpha=0.5)
plt.title('K Means Cluster Epicenters')

# Visualize clustes

plt.scatter(df1['Total Claim Amount'], df1['Monthly Premium Auto'], c=model.labels_)
plt.title('Total Claim Amount vs Monthly Premium')
plt.xlabel('Total Claim Amount')
plt.ylabel('Monthly Premium Auto')

plt.scatter(df1['Income'], df1['Monthly Premium Auto'], c=model.labels_)
plt.title('Income vs Monthly Premium')
plt.xlabel('Income')
plt.ylabel('Monthly Premium Auto')
plt.savefig('Income vs Monthly Premium', dpi=600)

# Visualize paired numerical columns with clustering model

for i in norm_cols:
    plt.show(plt.scatter(df[i], df['Customer Lifetime Value'], c=model.labels_))
    
for i in norm_cols:
    plt.show(plt.scatter(df[i], df['Income'], c=model.labels_))

for i in norm_cols:
    plt.show(plt.scatter(df[i], df['Monthly Premium Auto'], c=model.labels_))

for i in norm_cols:
    plt.show(plt.scatter(df[i], df['Months Since Policy Inception'], c=model.labels_))

for i in norm_cols:
    plt.show(plt.scatter(df[i], df['Number of Policies'], c=model.labels_))

for i in norm_cols:
    plt.show(plt.scatter(df[i], df['Months Since Last Claim'], c=model.labels_))
    
# Optimize K Means Model
    
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10)) # Cluster range 1-10, can be changed
visualizer.fit(X)
visualizer.poof()
visualizer.show(outpath="Elbow Kmeans Cluster.pdf")


## --- Agglomerative Cluster Model --- ##  

model = cluster.AgglomerativeClustering(n_clusters=3)
y_pred = model.fit_predict(X)
print(davies_bouldin_score(X,y_pred))
print(silhouette_score(X,y_pred))

# Optimize Models
for i in range(1,10):
    model = cluster.AgglomerativeClustering(n_clusters=i)
    y_pred = model.fit_predict(X)
    print('Number of Clusters ', i)
    print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
    print('Silhouette Score ', silhouette_score(X,y_pred))

metric_list = ['ward', 'complete', 'average', 'single']
for i in rmetric_list:
    model = cluster.AgglomerativeClustering(n_clusters=3, linkage=i)
    y_pred = model.fit_predict(X)
    print('Linkage is ', i)
    print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
    print('Silhouette Score ', silhouette_score(X,y_pred))
    
    
## --- DB Scan Cluster Model --- ##  

model = DBSCAN(eps=20, metric='euclidean', min_samples=6).fit(X)
y_pred = model.fit_predict(X)
print(davies_bouldin_score(X,y_pred))
print(silhouette_score(X, y_pred))

# Optimization

for i in range(1,10):
    clustering = DBSCAN(eps=i).fit(X)
    y_pred = clustering.fit_predict(X)
    print('Eps of ', i)
    print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
    print('Silhouette Score ', silhouette_score(X,y_pred))
    
metric_list = ['euclidean','manhattan']
for i in metric_list:
    clustering = DBSCAN(eps=6, metric=i).fit(X)
    y_pred = clustering.fit_predict(X)
    print('Metric ', i)
    print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
    print('Silhouette Score ', silhouette_score(X,y_pred))

for i in range(1,10):
    clustering = DBSCAN(eps=6, metric='euclidean', min_samples=i ).fit(X)
    y_pred = clustering.fit_predict(X)
    print('Min Samples ', i)
    print('Davies Bouldin Score ', davies_bouldin_score(X,y_pred))
    print('Silhouette Score ', silhouette_score(X,y_pred))
    
## --- All Models --- ##  
# Run all models and see the clustering when df is labeled
    
# KMeans
model = KMeans(n_clusters=3)
model.fit(X)
y_pred = model.predict(X)
print(davies_bouldin_score(X,y_pred))
print(silhouette_score(X, y_pred))
df['Cluster'] = model.labels_
df.Cluster.value_counts()


# DBSCAN
model = DBSCAN(eps=20, metric='euclidean', min_samples=6).fit(X)
y_pred = model.fit_predict(X)
print(davies_bouldin_score(X,y_pred))
print(silhouette_score(X, y_pred))
df['Cluster'] = model.labels_
df.Cluster.value_counts()


# Agglomeric
model = AgglomerativeClustering(n_clusters=3,linkage='ward', affinity='euclidean')
y_pred = model.fit_predict(X)
print(davies_bouldin_score(X,y_pred))
print(silhouette_score(X,y_pred))
df['Cluster'] = model.labels_
df.Cluster.value_counts()

## --- Visualization --- ##  

# KMeans -- best model > set cluster labels to df
model = KMeans(n_clusters=3)
model.fit(X)
y_pred = model.predict(X)
print(davies_bouldin_score(X,y_pred))
print(silhouette_score(X, y_pred))
df['Cluster'] = model.labels_
df.Cluster.value_counts()


# Explore the clustering model via pivot tables to gain insights 
pd.pivot_table(data=df,index='Cluster',values=norm_cols, aggfunc=('mean', 'count'))
pd.pivot_table(data=df,index=('Cluster', 'Policy Type'),values=norm_cols, aggfunc=('mean', 'count'))
pd.pivot_table(data=df,index=('Cluster', 'Coverage'),values=norm_cols, aggfunc=('mean', 'count'))
pd.pivot_table(data=df,index=('Cluster', 'Education'),values=norm_cols, aggfunc=('mean', 'count'))
# Make a list of only categorical columns
cat_columns = df.columns
cat_columns = [i for i in cat_columns if i not in  norm_cols]
# Take a look at clustering and income and education
df[['Income', 'Education']].groupby('Education').agg('mean').sort_values(by='Income', ascending=False)
# Look at how our clustering model is clustering income vs education in each cluster
df[['Income', 'Education', 'Cluster']].query('Cluster == 1').groupby('Education').agg('mean').sort_values(by='Income', ascending=False)
df[['Income', 'Education', 'Cluster']].query('Cluster == 2').groupby('Education').agg('mean').sort_values(by='Income', ascending=False)
df[['Income', 'Education', 'Cluster']].query('Cluster == 0').groupby('Education').agg('mean').sort_values(by='Income', ascending=False)

# Compare insights of normal dataset to clustering

# Avg income and education -- non clustering
df[['Income', 'Education']].groupby('Education').agg('mean').sort_values(by='Income', ascending=False)
avg_income= df[['Income', 'Education']].groupby('Education').agg('mean').sort_values(by='Income', ascending=False)
avg_income = avg_income.reset_index()
ax = sns.catplot(x="Education", y="Income", kind="bar", data=avg_income, aspect=1.5)
plt.title('Average Income By Education', weight='bold', size=14)
ax.set(ylabel='Avg Income')
plt.ylim(bottom=30000)
plt.savefig('Average Income By Education', dpi=600)

# Avg income and education with cluster 2
avg_income_cluster_2 = df[['Income', 'Education', 'Cluster']].query('Cluster == 2').groupby('Education').agg('mean').sort_values(by='Income', ascending=False)
avg_income_cluster_2=avg_income_cluster_2.reset_index()
ax = sns.catplot(x="Education", y="Income", kind="bar", data=avg_income_cluster_2, aspect=1.5)
plt.title('Average Income By Education', weight='bold', size=14)
ax.set(ylabel='Avg Income')
plt.ylim(top=55000)
plt.savefig('Average Income By Education by Cluster 2', dpi=600)

# Avg monthly Auto payment vs customer policy type -- non clustering
avg_monthly_auto = df[['Policy Type', 'Monthly Premium Auto']].groupby('Policy Type').agg('mean').sort_values(by='Monthly Premium Auto', ascending=False)
avg_monthly_auto = avg_monthly_auto.reset_index()
# Plot
ax = sns.catplot(x="Policy Type", y="Monthly Premium Auto", kind="bar", data=avg_monthly_auto, aspect=1.5)
plt.title('Average Monthly Premium Payment By Policy Type', weight='bold', size=14)
ax.set(ylabel='Avg Monthly Premium Payment')
plt.ylim(top=100)
plt.savefig('Average Monthly Premium Payment By Policy Type', dpi=600)

# Avg monthly Auto payment vs customer policy type -- All Clusters 

avg_monthly_auto_c = pd.pivot_table(data=df,index=('Cluster','Policy Type'),values='Monthly Premium Auto', aggfunc=('mean'))
avg_monthly_auto_c = avg_monthly_auto_c.reset_index()
#plot
ax = sns.catplot(x="Cluster", y="Monthly Premium Auto", kind="bar", hue='Policy Type', data=avg_monthly_auto_c, aspect=1.5)
plt.title('Average Monthly Premium Payment By Policy Type', weight='bold', size=14)
ax.set(ylabel='Avg Monthly Premium Payment')
plt.savefig('Average Monthly Premium Payment By Policy Type by Cluster', dpi=600)


## --- Prediction Model of Clustered Customers --- ## 

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Cluster', axis=1),
                                                   df.Cluster, test_size=1/3)
# List of models to test
model_list=[KNeighborsClassifier(), GaussianNB(), 
            DecisionTreeClassifier(), 
            RandomForestClassifier(), 
            CatBoostClassifier()]
# Loop through all models and test acc and prediction confusion matrix
l_acc = []
l_cm = []
for model in model_list:
    model2=model.fit(X=X_train, y=y_train)
    y_pred2 = model2.predict(X_test)
    l_acc.append(accuracy_score(y_test,y_pred2))
    l_cm.append(confusion_matrix(y_test,y_pred2))
    print(type(model2).__name__, ' is done')
# Show results
pd.DataFrame([[type(i).__name__ for i in model_list],l_acc]).T.sort_values(by=1)

# K Neighbors Model
model_KN = KNeighborsClassifier(n_neighbors=7)
res_KN = model_KN.fit(X_train,y_train)
pred_KN = model_KN.predict(X_test)
conf_KN = confusion_matrix(y_test,pred_KN)
print(conf_KN)
# RoC Curve
model_KN_roc = roc_auc_score(y_test,pred_KN)
fpr,tpr,thresholds = roc_curve(y_test, model_KN.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label=f'Model_KN (area={model_KN_roc})')
plt.plot([0,1], [0,1])
plt.legend()
plt.savefig('K Neighbors Model - RoC', dpi=600)
# Metrics
print(accuracy_score(y_test,pred_KN))
print(recall_score(y_test,pred_KN))
print(precision_score(y_test,pred_KN))
print(f1_score(y_test,pred_KN))
print(log_loss(y_test, pred_KN))















