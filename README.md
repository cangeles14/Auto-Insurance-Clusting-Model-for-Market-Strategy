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

Here is an image of our clustering model clustering customers and plotting that data with customers total claim amount  vs monthly payments. As we can see, our clusters are all distinct from one another and very few points overlap. This is a good visualization of clustering customers. 

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/totalclaimvsmonthlycluster.png" width="50%" height="75%">

Next we can take a look at how our clustering algorithm is clustering customers when we look at customers income and their monthly premium payment amount. We can see that the clustering algorithm fairs pretty well in this case. One thing to note is that we can see many datapoints with simply 0$ in income. This could be interfering with our model, and if we wanted we could rerun the model again and simply remove all customers who have 0$ income. This could allow the model to cluster without such large outliers in the data.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/Income%20vs%20Monthly%20Premium.png" width="50%" height="75%">

## Clustering Insights & Marketing Strategy

Next we want to utilize this model to look at trends within our clusters. This will allow us to optimize or produce a strong marketing campaign. I will demonstrate how creating a clustering algorithm will bring forth new insights on our customers and data that we would have notherwise not known of. Creating specific clusters for specific types of customers, we can more accurately target our customer and audience.

First I will take a look at our dataset and how we would normally classify and target those based on income and education.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/AvgIncomeVsEducationData.png" width="50%" height="75%">

In the above image you can see that as higher education is reached, there is an increase in the customers income. This is to be expected. Higher education usually means higher paying jobs.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/AvgIncomeVsEducationCluster2.png" width="50%" height="75%">

In this image, I looked at only one cluster, and examined these customers income and education. It looks very different. You can see that the same trend does not hold true. The average income is almost the same for all education levels. This tells us that if we were to target simply on education alone, we wouldn't be very accurate in our assumptions.

Next I will take a look at how my clustering model classifies customers based on thier monthly premium payments and the policy type they have.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/AvgMonthlyByPolicyData.png" width="50%" height="75%">

The above image is classification without utilizing the clustering algorithm. You can see that the average monthly payment is the same for all customers with differnent policy types. A marketing campaign targeting customers based on this information would simply target all customers.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/AvgMonthlyByPolicyClusters.png" width="50%" height="75%">

Next, if we take a look at the same data but this time clustering our customers, we can see that we infact have different groups of customers that have different monthly payments for each policy type. This allows us to target customers with those parameters. 

# Prediction Model

- Add our clustering algorithm back to our data and group customers by our new clusters
- Split our data into a training set and test set to create new model
- Test various model predictions 
- Run the best model and report on the accuracy of the model to predict new customers


Lastly, I took a look at how we can utilize this clustering model, combined with a prediction model, to target new incoming customers or predict and classify new customers that just join or prehaps are looking to join. This can not only strengthen our current marketing campaign, but also allow for onboarding of new customers.

Train/Test Split separates our data into training and test sets so we can build our prediction model. Next I create a list of models I wanted to test to see which prediction model had the highest accuracy.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/predictionmodelcode.png">

I then ran every model in the list and compared the accuracy of each model to choose which performed the best based on their accuracy.

<img src="https://github.com/cangeles14/Auto-Insurance-Clusting-Model-for-Market-Strategy/blob/master/visualizations/predictionmodelresults.png" width="30%" height="30%">

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
