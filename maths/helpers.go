package maths

import (
	"fmt"
	"math"
	"math/rand"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Tanh(x float64) float64 {
	e := math.Exp(2 * x)
	return (e - 1) / (e + 1)
}

// Standard Normal variate using Box-Muller transform.
func GaussianRandom(mean int, stdev int) float64 {
	u := 1 - rand.Float64() // Converting [0,1) to (0,1]
	v := rand.Float64()
	z := math.Sqrt(-2.0*math.Log(u)) * math.Cos(2.0*math.Pi*v)
	// Transform to the desired mean and standard deviation:
	return z*float64(stdev) + float64(mean)
}

func FloorInt(val float64) int {
	floored_val := math.Floor(val)
	if floored_val >= math.MaxInt64 || floored_val <= math.MinInt64 {
		panic(fmt.Sprintf("%+v is out of int range", val))
	}
	return int(floored_val)
}
