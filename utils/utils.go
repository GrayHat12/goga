package utils

import (
	"math"
	"math/rand"
	"sync"
)

type FilterCallback[T any] func(T, int, []T)

// -- Start REGION -- Copied from https://golangprojectstructure.com/removing-elements-from-slice-array/
func RemoveManyElementsByIndices[T any](slice []T, indices []int) []T {
	indicesMap := make(map[int]int)

	for _, index := range indices {
		indicesMap[index] = index
	}

	lastIndex := len(slice) - 1
	backIndex := lastIndex

	for _, index := range indices {
		if index < 0 || index > lastIndex {
			continue
		}

		mappedIndex := indicesMap[index]

		if mappedIndex == -1 {
			continue
		}

		if mappedIndex != backIndex {
			slice[mappedIndex] = slice[backIndex]

			indicesMap[backIndex] = indicesMap[mappedIndex]
		}

		indicesMap[index] = -1

		backIndex--
	}

	return slice[:backIndex+1]
}

func Filter[T any](slice []T, predicate func(T, int, []T) bool) []T {
	indices := make([]int, 0, len(slice))

	for index, element := range slice {
		if !predicate(element, index, slice) {
			indices = append(indices, index)
		}
	}

	return RemoveManyElementsByIndices(slice, indices)
}

// -- End REGION -- Copied from https://golangprojectstructure.com/removing-elements-from-slice-array/

func Find[T any](slice []T, predicate func(T, int, []T) bool) *T {
	for index, element := range slice {
		if predicate(element, index, slice) {
			return &element
		}
	}
	return nil
}

func ForEach[T any](slice []T, callback FilterCallback[T]) {
	wg := sync.WaitGroup{}
	callbackWrapper := func(waitGroup *sync.WaitGroup, callback func()) {
		defer waitGroup.Done()
		callback()
	}
	for index, element := range slice {
		wg.Add(1)
		go callbackWrapper(&wg, func() {
			callback(element, index, slice)
		})
	}
	wg.Wait()
}

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
