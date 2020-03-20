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

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/DataScaling.png">

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/Customer%20Lifetime%20Value%20Distribution.png">

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/Income%20Distribution.png">

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
