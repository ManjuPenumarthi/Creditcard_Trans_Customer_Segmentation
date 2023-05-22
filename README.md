# Customer-segmentation
This project is to develop an analytical model for Redstone Federal Credit Union (RFCU) that segments their customers into different categories based on their daily transaction data. The model will provide insights and recommendations to inform RFCU's decisions related to credit line management, marketing, and fraud detection.

# Project Objectives
The main objective of this project is to segment Redstone Federal Credit Union's customers into different categories based on their transaction patterns. This will be achieved by conducting an analysis of the transaction data and identifying accounts with similar spending behavior. The data will be evaluated based on transaction amount, frequency, merchant, location, and seasonality.

Using Python programming, the team will implement various clustering algorithms to efficiently identify the different customer segments. These segments may include
high-spending customers, low-frequency/high-value customers, seasonal spenders, channel-specific spenders, and category-specific spenders, among others.

The final outcome of this project will be a set of recommendations for RFCU on which customer segments to target for various financial purposes. These recommendations will be
based on the results of the clustering analysis and will be presented to the RFCU team for their consideration and implementation.

# Data Summary
# Initial Data
Our initial data consisted of 16 million rows of customer transactions made using RFCU cards.
The features include:
NEW_MCID: Unique Customer Identifier
RDT_MRCH_SIC_CODE: Merchant Industry Code
RDT_TRANSACTION_AMOUNT: Transaction Amount
TRX_DATE: Transaction Date
RDT_TRANSACTION_CODE: Transaction Code
RDT_CHD_EXT_STATUS: External Authorization Status
RDT_CHD_INT_STATUS: Internal Authorization Status
RDT_MERCHANT_CITY: Merchant City
RDT_MERCHANT_NAME: Merchant Name
RDT_MERCHANT_STATE: Merchant State
BIN: Bank Identification Number
In total, there were around 85k customers analyzed in our project.

# Data Modeling:
We applied four different clustering algorithms to identify a similar group of customers.

**K-means Clustering:**
K-means clustering is a popular unsupervised machine learning algorithm used to partition a given dataset into K clusters or groups, based on the similarity of the data points within each cluster. The goal of the algorithm is to minimize the sum of squared distances between the data points and their assigned cluster centroids.
The algorithm works as follows:
1. Choose the number of clusters, K, that you want to partition the dataset into.
2. Randomly initialize K cluster centroids.
3. Assign each data point to the cluster whose centroid is closest to it.
4. Recalculate the centroids of each cluster as the mean of the data points assigned to it.
5. Repeat steps 3 and 4 until the algorithm converges, i.e., the cluster assignments no longer change.
6. Finally, the algorithm returns the K cluster centroids and the assignment of each data point to a particular cluster.

**Silhouette Score to find the best number of clusters in K-means:** After testing potential K values between 3 and 10; the higher silhouette score is shown at K=5. So, we chose 5 as the best value for K in this application in the K-means algorithm.

**Mean Shift Clustering:**
Mean shift clustering is another popular unsupervised machine learning algorithm used for clustering similar data points together. It works by finding the mode or the maximum density region of a probability density function, which represents the data distribution of the input data points.
The algorithm works as follows:
1. Initialize each data point as a cluster centroid.
2. For each data point, compute the mean shift vector by taking the weighted average of the difference between the data point and its neighboring points, where the weights are given by a kernel function (such as Gaussian kernel).
3. Update each data point to be the point to which the mean shift vector points, which is usually towards the maximum density region of the probability density function.
4. Merge clusters that are close to each other by setting their centroids to be the average of the data points in both clusters.
5. Repeat steps 2-4 until the centroids no longer move or the algorithm converges.
6. Finally, the algorithm returns the cluster assignments of each data point.

**Gaussian Mixture Clustering:**
Gaussian Mixture Clustering is a popular unsupervised machine learning algorithm used to model the probability distribution of a dataset using a mixture of Gaussian distributions. It works by estimating the parameters of the Gaussian distributions that best fit the data and then assigning each data point to the Gaussian distribution with the highest probability.
The algorithm works as follows:
1. Initialize the parameters of the Gaussian mixture model, such as the number of components, mean, and covariance matrix of each component.
2. E-Step: Assign each data point to a component by computing the posterior probability of the data point belonging to each component using Bayes' rule and the current model parameters.
3. M-Step: Update the model parameters, such as the mean and covariance matrix of each component, using the data points assigned to that component.
4. Repeat steps 2 and 3 until the algorithm converges or a stopping criterion is met.
5. Finally, the algorithm returns the model parameters and the cluster assignments of each data point.

**BIRCH Clustering:**
BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is a hierarchical clustering algorithm used to cluster large datasets. It was designed to be memory-efficient and able to handle large datasets without having to store all data points in memory. Instead, BIRCH builds a tree-based data structure, known as a Clustering Feature Tree (CFT), which allows for the efficient clustering of large datasets.
The algorithm works as follows:
1. Initialize the CFT with a user-defined threshold value for the maximum number of data points that can be stored in each leaf node.
2. For each data point, insert it into the CFT by updating the corresponding leaf node's parameters, such as the centroid and the number of data points in the node.
3. If a leaf node's number of data points exceeds the threshold value, split the node into two new child nodes and redistribute the data points between them.
4. Merge nodes that are close to each other and satisfy a user-defined merging criterion.
5. Repeat steps 2-4 until all data points are inserted into the CFT.
6. Finally, traverse the CFT to generate the hierarchical clusters.
