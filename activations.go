package phase

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

	"smooth_relu": SmoothReLU,
	"wavelet_act": WaveletAct,
	"cauchy_act":  CauchyAct,
	"asym_act":    func(x float64) float64 { return AsymAct(x, 0.1) },
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
func (bp *Phase) InitializeActivationFunctions() {
	bp.ScalarActivationMap = InitializeActivationFunctions()
}

// SmoothReLU activation function
func SmoothReLU(x float64) float64 {
	return x / (1 + math.Exp(-x))
}

// ParamReLU activation function
func ParamReLU(x, a, b float64) float64 {
	if x > 0 {
		return a * x
	}
	return b * (-x)
}

// WaveletAct activation function
func WaveletAct(x float64) float64 {
	return (1 - x*x) * math.Exp(-x*x/2)
}

// AsymAct activation function with fixed a
func AsymAct(x, a float64) float64 {
	if x > 0 {
		return x
	}
	return a * x * x
}

// CauchyAct activation function
func CauchyAct(x float64) float64 {
	return (1/math.Pi)*math.Atan(x) + 0.5
}
