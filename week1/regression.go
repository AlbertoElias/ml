package main

import (
	"fmt"
	"math"
)

type Data struct {
	x float64
	y float64
}

func GenerateFakeData() []Data {
	data := make([]Data, 3)
	for i := 0; i < len(data); i++ {
		data[i] = Data{
			x: float64(i),
			y: float64(i),
		}
	}

	return data
}

func Hypothesis(theta0 float64, theta1 float64, x float64) float64 {
	return theta0 + theta1*x
}

func GradientDescent(dataset []Data) (float64, float64) {
	const alpha float64 = 0.05
	var theta0 float64 = 0.0
	var theta1 float64 = 0.0
	var m float64 = float64(len(dataset))

	for {
		var theta0DataSum float64 = 0.0
		for i := 0; i < len(dataset); i++ {
			theta0Hypothesis := Hypothesis(theta0, theta1, dataset[i].x) - dataset[i].y
			theta0DataSum = theta0DataSum + theta0Hypothesis
		}
		var temptheta0 float64 = theta0 - alpha*(1.0/m)*theta0DataSum

		var theta1DataSum float64 = 0.0
		for i := 0; i < len(dataset); i++ {
			theta1Hypothesis := (Hypothesis(theta0, theta1, dataset[i].x) - dataset[i].y) * dataset[i].x
			theta1DataSum = theta1DataSum + theta1Hypothesis
		}
		var temptheta1 float64 = theta1 - alpha*(1.0/m)*theta1DataSum

		if math.Abs(theta0-temptheta0) <= 0.000001 || math.Abs(theta1-temptheta1) <= 0.000001 {
			break
		}

		theta0 = temptheta0
		theta1 = temptheta1
	}

	return theta0, theta1
}

func mainWeek1() {
	theta0, theta1 := GradientDescent(GenerateFakeData())
	fmt.Println(fmt.Sprintf("The value for 3 with theta0 %.6f and theta1 %.6f is: %.6f", theta0, theta1, Hypothesis(theta0, theta1, 3)))
}
