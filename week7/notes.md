# Support Vector Machines (SVMs)

Supervised machine learning algorithm

We take the logistic regression cost function but we modify both terms:

* OTx >> 1, y = 1 (cost1(z))
* OTx << -1, y = 0 (cost0(z))

z = OTx

This leaves us with this cost function:

J(θ) = (1 / m) * ∑ y(i) * cost1(θTx(i)) + (1 − y(i)) * cost0(θTx(i)) + λ * 2 * m * ∑ Θj ^ 2

We can simplify this by multipling by m (which doesn't alter the cost value) and by regularizing with a constant C instead of λ (C = 1 / λ):

J(θ) = C * ∑ y(i) * cost1(θTx(i)) + (1 − y(i)) * cost0(θTx(i)) + (1 / 2) * m * ∑ Θj ^ 2

To regularize more, we decrease C

vhθ(x) isn't the probability of y being 1 or 0, in SVM, it outputs either

## Intuition

SVMs are _Large Margin Classifiers_

If C is very large, ∑ y(i) * cost1(θTx(i)) + (1 − y(i)) * cost0(θTx(i)) needs to equal 0 which reduces J(θ) to:

J(θ) = (1 / 2) * m * ∑ Θj ^ 2

The decision boundary in SVM is as far away as possible from positive and negative examples. The distance between it and the nearest example is called *margin*, and SVMs try to maximize it

To increase the margin, we increase C. If there are outlier examples, and we don't want them to affect the decision boundary, we reduce C

## Maths

Using vector inner products:

u * v = uT * v = p * ||u||

p = projection of vector v onto vector u

minΘ (1 / 2) * ∑ Θj ^ 2 =
    (1 / 2) * (Θ1 ^ 2 + Θ2 ^ 2 + … + Θn ^2) =
    (1 / 2) * (sqrt(Θ1 ^ 2 + Θ2 ^ 2 + … + Θn ^2)) ^ 2 =
    (1 / 2) * ∣∣Θ∣∣ ^ 2

We can also rewrite z:

ΘTx(i) = p(i) * ∣∣Θ∣∣ = Θ1 * x(i)1 + Θ2 * x(i)2 + … + Θn * x(i)n

So now we have:

* If y = 1, we want p(i) * ∣∣Θ∣∣ ≥ 1
* If y = 0, we want p(i) * ∣∣Θ∣∣ ≤ −1

This causes a "large margin" because the vector for Θ is perpendicular to the decision boundary. In order for our optimization objective (above) to be true, we need p(i) to be as large as possible.

## Kernels

Used to make non-linear, complex classifiers using SVMs, as it wouldn't be efficient with other algorithms

fi = similarity(x, l(i)) = exp(− (∣∣x − l(i)∣∣ ^ 2) / (2 * (σ ^ 2))) =
exp(− (∑(xj − l(i)j) ^ 2) / (2 * (σ ^ 2)))

We compute a feature (fi) by finding the similarity between x and a landmark l and this similarity function is called a *Gaussian kernel*

Properties:

* If x ≈ l(i), then fi = exp(− (≈0 ^ 2) / (2 * (σ ^ 2))) ≈ 1
* If x is far from l(i), then fi = exp(− (large number ^ 2) / (2 * (σ ^ 2))) ≈ 0

Landmarks are features in the hypothesis:

hΘ(x) = Θ1 * f1 + Θ2 * f2 + Θ3 * f3 + …

σ2 is a parameter of the Gaussian Kernel. It can be modified to increase or decrease how fast we get to fi

### Pick landmarks

Same locations as the training examples, so we have m landmarks. We may also set f0 = 1 to correspond with and we get a feature vector f(i) of all our features for example x(i)

To get Θ, we use SVM like this:

![Kernel SVM](images/kernel_svm.png)

### Picking parameters

C:

* If C is large, then we get higher variance/lower bias
* If C is small, then we get lower variance/higher bias

σ2:

* Large σ2, the features fi vary more smoothly, causing higher bias and lower variance
* Small σ2, the features fi vary less smoothly, causing lower bias and higher variance

## Using an SVM

* Pick C
* Pick Kernel
    - No kernel (linear kernel). When n is large, m is small
    - Gaussian kernel. Pick σ2. When n is small, m is large
        + Feature scaling

SVM algorithms must satisfy _Mercer's theorem_ so that the optimizations run correctly

Parameters of the kernel are trained in the training and the cross validation sets

## Multiclass classification

*one vs all*

y ∈ 1 , 2 , 3 , … , K with Θ(1), Θ(2), …, Θ(K)

We pick class i with the largest (Θ(i))T * x

## Logistic regression vs SVM

* n is large (relative to m): Logistic regression or linear kernel
* n is small and m is intermediate: SVM with a Gaussian kernel
* n is small and m is large: Add more features, then logistic regression or linear kernel

A neural network is likely to work well for any of these situations, but may be slower to train

     
