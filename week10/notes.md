# Large scale

Check that a higher number of examples will offer better performance by plotting the _learning curve_ with a small number of examples. If it has high variance, you need more examples

## Stochastic gradient descent

cost(θ, (x(i), y(i))) = (1 / 2) * (hθ(x(i)) − y(i)) ^ 2

We've removed the constant _m_

Jtrain(θ) = (1 / m) ∑i=1:m cost(θ, (x(i), y(i)))

Steps:

1. Shuffle training examples
2. Repeat:
for i:=1, ..., n
    θj := θj − α(hΘ(x(i)) − y(i)) * x(i)j     * for every j:=0, ..., n

This way, it starts fitting θj with just one example

It usually takes 1-10 passes to get close to the global minimum

It doesn't converge in the global minimum, it wanders around it

### Checking convergence

1. Calculate cost for (x(i), y(i)) in every iteration before updating θ
2. Every 1000 iterations, calculate average cost and plot it

Smaller α may improve the result as it reduces the oscillation around the global minimum. To actually converge at the global minimum, α could be reduced over time: α = const1i / (terationNumber + const2) but it means fiddling with more parameters

More examples to calculate average cost => smoother line

## Mini-batch gradient descent

Uses a smaller batch size _b_ in each iteration (usually 2-100)

We use more than one example to take advantage of vectorisation, which is an advantage over _Stochastic gradient descent_

## Online learning

Continuos stream of incoming data

You're continuously updating θ for each user (x, y) as we collect data from them

Usage:

* Predicted click through rate (CTR)
* Customizing news article
* Product recommendations
* Special offers for users


## Map reduce

We divide batch gradient descent into _z_ subsets, each running on different machines, calculating the cost function for each one

MapReduce takes all these summations, and reduces them by calculating:

Θj := Θj − α * (1/z) * (temp(1)j + temp(2)j + ⋯ + temp(z)j)

We can apply to:

* Linear regression
* Logistic regression
* Neural networks: for back and forward propagation
