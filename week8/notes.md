# Unsupervised learning

Unlabeled training set. Good for:

* Market segmentation
* Social network analysis
* Organizing computer clusters
* Astronomical data analysis

## K-means algorithm

Clustering algorithm:

1. Randomly initializes K _cluster centroids_ μ(1), μ(2), ..., μ(K)
2. Repeat
    2.1 for i = 1 to m
            c(i) := index (from 1 to K) of closest cluster centroid to x(i)
    2.2 for i = 1 to K
            μ(k) := average of points assigned to K

* K: Number of clusters
* Training set x(1), x(2), ..., x(m) where x(i) ∈ ℝ ^ n
* The first for loop assigns the c vector where c(i) is the centroid assigned to x(i)
    - c(i) = min(k) ∣∣ x(i) − μk ∣∣ ^ 2
* The second for loop is to move the centroid to the average of its group
    - μk = 1 / n [x(k1) + x(k2) + … + x(kn)] ∈ ℝ ^ n
    - x(kn) are the training examples assined to μk
    - If there are no points assigned to a centroid, it can be randomly reinitialised
* After a number of iterations, it will converge and clusters won't be changed

### Cost function

μc(i): cluster centroid of cluster to which example x(i) has been assigned

J(c(i), …, c(m), μ1, …, μK) = 1 / m ∑∣∣x(i) − μc(i)∣∣ 2

Also called _distortion_ of the training examples

The objective is `minc,μ J(c,μ)`

We are finding all the values in sets c _clusters_, and μ _centroids_, that will minimize the average of the distances of every training example to its cluster centroid

In the _cluster assignment_ step, we look to:

Minimize J(…) with c(1), …, c(m) (holding μ1, …, μK fixed)

In the _move centroid_ step:

Minimize J(…) with μ1, …, μK

*The cost function can never increase*

### Random initialization

Recommended method:

1. K must be less than m
2. Randomly pick K training examples
3. Set μ1, …, μk equal to these K examples

K-means could get stuck in a local optima, so the algorithm should be ran on different random initializations. You then compute the cost function for each execution and pick the clustering with the lowest cost

### Picking number of clusters

*Elbow method*

We plot the cost function and the number of clusters. The cost function reduces while we increase the clusters until it flattens. We pick K at that point

Sometimes, the _elbow_ isn't clear

Another way, is to pick a K that makes sense downstream

## PCA - Principal Component Analysis

Reduces dimension of features:

* If we have a lot of redundant data
* Optimization, reduces total amount of data
* Visualization, as we can only display data in up to 3 dimensions.

### Algorithm

Reduce from n-dimension to k-dimension: Find k vectors u(1), u(2), …, u(k) onto which to project the data so as to minimize the projection error

* If we go from 2D to 1D, we map are 2 features to a line
* If we go from 3d to 2D, we map those 3 features to a plane
* In PCA, we search for the shortest distance between the data points (orthogonal distance). In linear regression, we search for smallest squared error between each point and the predictor line (vertical distance)

We want: x(i) ∈ ℝ ^ n => z(i) ∈ ℝ ^ k
1. Perform mean normalization and feature scaling
2. Compute _covariance matrix_ Σ
    2.1. Σ = (1 / m) * ∑ x(i) * x(i)' 
    2.2. x(i): n x 1
    2.3. x(i)': 1 x n
    2.4. X: m x n
    2.5. Σ: n x n
3. Compute _eigenvectors_ of Σ
    3.1. `[U,S,V] = svd(Sigma);`
    3.2. `svd()` is _singular value decomposition_
    3.3. U ∈ ℝ ^ (n×n)
4. Take first *k* columns of U to compute *z*
    4.1. z(i) = Ureduce' * x(i)
    4.2. Ureduce: k x n
    4.3. z(i): k x 1

It should only be applied on the training set, not on the cross validation or the testing set

It shouldn't be used for overfitting, use regularization. PCA doesn't take *y* into account

Always try running learning algorithm without PCA first, only use PCA if needed

#### Picking k

Average squared projection error:

(1 / m) * ∑ ∣∣ x(i) − x(i)approx ∣∣ ^ 2

Total variation of data:

(1 / m) * ∑ ∣∣ x(i) ∣∣ ^ 2

Pick the smallest _k_ such that:

*Average squared projection error* / *Total variation* <= 0.01%

This way, 99% of variance is retained

We start with k = 1, and run PCA, increasing k by one in each iteration, until a value of k retains enough variance

This isn't very efficient, so, in Octave, we can use the matrix _S_ returned by `svd()` and use it like this:

∑(k, i=1) Sii / ∑(n, i=1) Sii >= 0.99

#### Reconstruction

z(i) ∈ ℝ ^ k => x(i) ∈ ℝ ^ n

x(i)approx = Ureduce * z(i)