package main

import (
	"fmt"
	"math"
)

type Data struct {
	x []float64
	y float64
}

func GenerateFakeData() []Data {
	data := make([]Data, 3)
	for i := 0; i < len(data); i++ {
		data[i] = Data{
			x: []float64{float64(i), float64(i), float64(i)},
			y: float64(i),
		}
	}

	return data
}

func AdjustX(originalx []float64) []float64 {
	x := []float64{1}
	x = append(x, originalx...)
	return x
}

func Hypothesis(theta []float64, x []float64) float64 {
	if len(theta) != len(x) {
		panic("Vectors theta and x need to have same amount of values")
	}

	result := 0.0
	for i := 0; i < len(theta); i++ {
		result = result + theta[i]*x[i]
	}
	return result
}

func Cost(theta []float64, dataset []Data) float64 {
	var m float64 = float64(len(dataset))

	var thetaDataSum float64 = 0.0
	for i := 0; i < len(dataset); i++ {
		x := AdjustX(dataset[i].x)
		thetaHypothesis := math.Pow(Hypothesis(theta, x)-dataset[i].y, 2)
		thetaDataSum = thetaDataSum + thetaHypothesis
	}

	return thetaDataSum / (2.0 * m)
}

func GradientDescent(dataset []Data) []float64 {
	const alpha float64 = 0.3
	var m float64 = float64(len(dataset))
	var theta []float64 = make([]float64, len(dataset[0].x)+1)

	// Populate theta with initial values of 0
	for j := 0; j < len(theta); j++ {
		theta[j] = 0
	}

	var currentCost float64
	for {
		var tempTheta []float64 = make([]float64, len(theta))
		for t := 0; t < len(theta); t++ {
			var thetaDataSum float64 = 0.0
			for i := 0; i < len(dataset); i++ {
				x := AdjustX(dataset[i].x)
				thetaHypothesis := (Hypothesis(theta, x) - dataset[i].y) * x[t]
				thetaDataSum = thetaDataSum + thetaHypothesis
			}
			tempTheta[t] = theta[t] - alpha*thetaDataSum/m
		}

		cost := Cost(tempTheta, dataset)
		if math.Abs(currentCost-cost) <= 0.00001 {
			break
		}

		currentCost = cost
		theta = tempTheta
	}

	return theta
}

func main() {
	fake := GenerateFakeData()
	fmt.Println("Fake data: ", fake)
	theta := GradientDescent(fake)
	x := AdjustX([]float64{3, 3, 3})
	fmt.Println(fmt.Sprintf("The value for 3 with theta0 %.6f, theta1 %.6f, theta2 %.6f and theta3 %.6f is: %.6f", theta[0], theta[1], theta[2], theta[3], Hypothesis(theta, x)))
}
