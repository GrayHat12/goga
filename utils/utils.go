package utils

import (
	"math"
	"math/rand"
)

type GaussianRandomInput struct {
	Mean  float64
	Stdev float64
}

func GaussianRandom(params *GaussianRandomInput) float64 {
	if params == nil {
		params = &GaussianRandomInput{Mean: 0, Stdev: 1}
	}
	u := 1 - rand.Float64() // Converting [0, 1) to (0, 1]
	v := rand.Float64()
	z := math.Sqrt(-2*math.Log(u)) * math.Cos(2*math.Pi*v)

	// Transform to the desired mean and standard deviation
	return z*params.Stdev + params.Mean
}

func Tanh(x float64) float64 {
	e := math.Exp(2 * x)
	return (e - 1) / (e + 1)
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
