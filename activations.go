package blueprint

import "math"

// ActivationFunc defines the type for scalar activation functions
type ActivationFunc func(float64) float64

// Supported scalar activation functions
var scalarActivationFunctions = map[string]ActivationFunc{
	"relu":       ReLU,
	"sigmoid":    Sigmoid,
	"tanh":       Tanh,
	"leaky_relu": LeakyReLU,
	"elu":        ELU,
	"linear":     Linear,
}

// ReLU activation function
func ReLU(x float64) float64 {
	return math.Max(0, x)
}

// Sigmoid activation function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Tanh activation function
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// LeakyReLU activation function
func LeakyReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}

// ELU activation function
func ELU(x float64) float64 {
	if x >= 0 {
		return x
	}
	return 1.0 * (math.Exp(x) - 1)
}

// Linear activation function
func Linear(x float64) float64 {
	return x
}

// InitializeActivationFunctions returns the activation functions map
func InitializeActivationFunctions() map[string]ActivationFunc {
	return scalarActivationFunctions
}

// InitializeActivationFunctions initializes the ScalarActivationMap
func (bp *Blueprint) InitializeActivationFunctions() {
	bp.ScalarActivationMap = InitializeActivationFunctions()
}
