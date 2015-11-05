# Linear Regression with > 1 Variable

"multivariate linear regression". Single ouput from multiple inputs.

## Hypothesis Function

θ and x are vectors with n + 1 rows where x0 = 1

hθ (x) = θT * x = θ0 * x0 + θ1 * x1 ... θn * xn

T = transpose

## Cost Function

Now we just take the vector θ

J(θ) = (1 / (2 * m)) * ∑(hθ(xi) - yi) ^ 2

## Gradient Descent

θj := θj - α * derivative of J(θ)

Adding the derivative of θj:

θj := θj - α * (1 / m) * ∑((hθ(xi) - yi) * x(j)i)

### Feature scaling

To make sure features are on a similar scale, divide the feature by the maximum value in its range. Eg. size (0-2000 feet^2): size / 2000

It gets every feature into an approximately -1 <= xi <= 1 range

#### Mean normalization

Replace xi with xi - μi. (not to x0 = 1) μi is the average value of the feature in training set. Eg. size(0-2000: 1000)

We divide by range (maximum value - minimum value)

Range: -0.5 <= xi <= 0.5

### Convergence test

Converged if J(θ) decreases by less than some small value (i.e. 10^-3) in an iteration

### Learning rate

- If α is too large: may not converge or slow convergence
- If α is too small: slow convergence

Try 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1 (~3x)

### Polynomial Regression

Feature: size

hθ(x) = θ0 + θ1(size) + θ2(size)^2 + θ3(size)^3
hθ(x) = θ0 + θ1(size) + θ2(sqrt(size))

## Normal equation

Gets the value of θ directly by:

Derivative of J(θ) = 0

Equation:

θ = ((XT * X) ^ (−1)) * XT * y

- No need for feature scaling
- No need for alpha
- No iterations
- Slow if n is large. Calculating inverse is O(n ^ 3). When n > 10000, gradient descent

### Proof

1. θ is a (n+1)x1 matrix
2. X is a mx(n+1) matrix so XT is a (n+1)xm matrix
3. Y is a mx1 matrix
4. Xθ = Y The point is to inverse X, but as X is not a square matrix we need to use XT to have a square matrix
5. (XT) * X * θ = (XT) * Y
6. Associative matrix multiplication: (XT * X)* θ = XT * Y
7. Assuming (XT * X) invertible: θ = ((XT * X) ^ (−1)) XT * y

### Noninvertability

XT * X could be noninvertible due to:

- Redundant features (they are linear dependent). Delete them
- Too many features (m < n). Delete some or use "regularization"
