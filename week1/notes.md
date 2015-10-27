Machine learning: "A computer is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

# Supervised learning

Relationship between input and output

- Regression. Map input variables to some continuous function
- Classification. Map input variables into discrete categories

# Unsupervised learning

Derive structure from data

- Clustering. Puts together data based on relationships among the variables in the data.
- Associative. Estamates mappings between variables.

# Linear Regression with One Variable

"univariate linear regression". Single ouput from single input.

## Hypothesis Function

hθ (x) = θ0 + θ1 * x

## Cost Function

Measures the accuracy of the hypothesis function taking the average of all the results of the hypothesis with all the inputs

J(θ0, θ1) = (1 / (2 * m)) * ∑(hθ(xi) - yi) ^ 2

## Gradient Descent

Improves the hypothesis function. It finds the local minimum in the graph of the cost function against θ0 and θ1.

We take the derivative of the cost function, which gives us the direction to move towards at a learning rate α.

θj := θj - α * derivative of J(θ0, θ1)

Applied to θ0 and θ1:

θ0 := θ0 - α * (1 / m) * ∑(hθ(xi) - yi)

θ1 := θ1 - α * (1 / m) * ∑((hθ(xi) - yi) * xi)

