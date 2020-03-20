# Auto Insurance Market Strategy Optimization with Clustering Algorithm
Using a clustering machine learning model, I created a classification algorithm that accurately targets clients, increasing the strength of a market strategy

## Objectives

- Optimize the marketing campaign by categorizing its clientele

- Construct a classification algorithm that can be used for the new marketing strategy

- Accurately target clients for specific needs, increasing the strength of its marketing campaign

## Roadmap

- Data Analysis: Utilizing python, pandas, and dataset provided by kaggle.com
- Data Processing: Data beautifying through data reconfiguration and scaling
- Model Building: Create an unsupervised model using the sklearn library to cluster our data
- Metrics: Put our model through various optimization parameters

## Understand the Data

One of the most important steps before analyzing any dataset is identifying which factors are most important to the problem at hand. This helps us build a model to accurately classify customers.

Important numerical and categorical factors including:
- Education
- Coverage
- Monthly Premium Payments
- Policy Type

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/dataset.png">

## Data Manipulation, Feature Engineering & Data Preprocessing

Preparing the dataset to reflect what insights we want to extract from it is key to gathering insightful information from our data and to prevent unwanted bias

By looking at our data distributions, we can see which variables have outliers or are not normally distributed. From there we can clean and process our dataset by:

- Dropping columns
- Changing columns from numerical to categorical
- Windorizing and standardizing numerical columns

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/datasetProcessing.png">

Scaling our data allows us to keep the same distribution of our dataset while allowing us to compare two variables that are seeminly unrelated.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/DataScaling.png" width="50%" height="75%">

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/Customer%20Lifetime%20Value%20Distribution.png" width="50%" height="75%">

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/Income%20Distribution.png" width="50%" height="75%">

## Building the Model

- Create an unsupervised machine model that can cluster our dataset customers
- Visualize these clusters and analyze similarities within clusters 
- Calculate how well our model is clustering our dataset
- Test  how well our model can predict new data points

The three clustering methods I will perform are K Means Clustering, DB Scan Clustering, and lastly, Agglomerative Clustering.

* [K Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - Aims to partition observations into clusters in which each observation belongs to the cluster with the nearest mean

* [DB Scan](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) - Density based, spatial clustering method that groups together points that are closely packed together

* [Agglomerative](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) - Also called hierarchical clustering, a bottom-up clustering approach where each observation is assigned its own cluster and by calculating the similarity between clusters, merges them

## Model Performance & Metrics

Optimization allows our model to perform at its maximum to classify clusters. I will go over the models and some of the methods I used at optimizing them and analyzing their metrics.

Metrics:

* [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) - Silhouette Score is a classification metrics that measures how similar a datapoint is to its own cluster compared to all clusters

* [Dunn Index]() - Dunn Index uses cluster size and intercluster distances to evaluate clustering algorithms

* [Davies-Bouldin Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html) - Davies-Bouldin Index uses the ratio of parameters within the cluster to the parameters between clusters to optimize clustering algorithms 

K Means is an algorithm that tries to partition the dataset into distinct, non-overlapping groups or clusters. After running the model we can plot the epicenters of our clusters. This helps tell us if our clusters are distinct or not.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/KMeansClusteringCenters.png" width="50%" height="75%">

K Means Elbow is a visualization method that computes the optimal number of clusters for a K Means model within the ranges specified

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/KElbowVis.png">

Looping through the metrics for DB Scan and Agglomerative clustering models, we can find the optimal parameters for each model. 

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/DBAggloOptimization.png">

## Visualization of Clusters

One of the most important things when creating a clustering model is to actually visualize your clusters with your dataset. It's almost impossible to tell if your clustering algorithm is comparively good simply on metric scores.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/Monthly%20Premium%20vs%20Total%20Claim.png" width="50%" height="75%">

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/Income%20vs%20Monthly%20Premium.png" width="50%" height="75%">


## Presentation

* [Presentation](https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/Auto%20Insurance%20Clustering%20Model.pdf)

## Built With

* [Python](https://docs.python.org/3/) - The programming language used
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language
* [MySQL](https://www.mysql.com/) -  MySQL is an open-source relational database management system for SQL
* [Tableau](https://www.tableau.com/) - Popular Data visualization tool
* [MatPlotLib](https://matplotlib.org/contents.html) - Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms

## Authors

* **Christopher Angeles** - [cangeles14](https://github.com/cangeles14)

## Acknowledgments

* [Ironhack](https://www.ironhack.com/en/data-analytics) -  Data Analytics Bootcamp @ Ironhack Paris
