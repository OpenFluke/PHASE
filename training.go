package phase

import (
	"math"
)

// TrainNetwork trains the network using backpropagation.
func (bp *Phase) TrainNetwork(inputs map[int]float64, expectedOutputs map[int]float64, learningRate float64) {
	// Set a reasonable learning rate to prevent instability
	if learningRate <= 0 || learningRate > 0.1 {
		learningRate = 0.001 // Default to a small, stable value
	}

	// Forward pass
	bp.Forward(inputs, 1)

	// Compute output errors
	outputErrors := make(map[int]float64)
	for id, expected := range expectedOutputs {
		actual := bp.Neurons[id].Value
		outputErrors[id] = expected - actual
	}

	// Backward pass
	for id, neuron := range bp.Neurons {
		if neuron.Type == "input" {
			continue
		}
		errorTerm := 0.0
		if err, isOutput := outputErrors[id]; isOutput {
			errorTerm = err * bp.activationDerivative(neuron.Value, neuron.Activation)
		} else {
			for _, downstreamID := range bp.getDownstreamNeurons(id) {
				if downstreamErr, exists := outputErrors[downstreamID]; exists {
					weight := bp.getWeight(id, downstreamID)
					errorTerm += downstreamErr * weight
				}
			}
			errorTerm *= bp.activationDerivative(neuron.Value, neuron.Activation)
		}

		// Update weights and bias with NaN checks and clamping
		for i, conn := range neuron.Connections {
			sourceID := int(conn[0])
			sourceValue := bp.Neurons[sourceID].Value
			gradient := errorTerm * sourceValue
			if !math.IsNaN(gradient) && !math.IsInf(gradient, 0) {
				neuron.Connections[i][1] += learningRate * gradient
				// Clamp weight to [-5, 5]
				if neuron.Connections[i][1] > 5 {
					neuron.Connections[i][1] = 5
				} else if neuron.Connections[i][1] < -5 {
					neuron.Connections[i][1] = -5
				}
			}
		}
		if !math.IsNaN(errorTerm) && !math.IsInf(errorTerm, 0) {
			neuron.Bias += learningRate * errorTerm
			// Clamp bias to [-5, 5]
			if neuron.Bias > 5 {
				neuron.Bias = 5
			} else if neuron.Bias < -5 {
				neuron.Bias = -5
			}
		}
	}
}

// activationDerivative computes the derivative of the activation function.
func (bp *Phase) activationDerivative(value float64, activation string) float64 {
	switch activation {
	case "sigmoid":
		sig := Sigmoid(value)
		return sig * (1 - sig)
	case "relu":
		if value > 0 {
			return 1
		}
		return 0
	case "tanh":
		t := Tanh(value)
		return 1 - t*t
	case "leaky_relu":
		if value > 0 {
			return 1
		}
		return 0.01
	case "elu":
		if value >= 0 {
			return 1
		}
		return ELU(value)
	case "linear":
		return 1
	default:
		return 1
	}
}

// getDownstreamNeurons finds neurons that depend on the given neuron.
func (bp *Phase) getDownstreamNeurons(neuronID int) []int {
	downstream := []int{}
	for id, neuron := range bp.Neurons {
		for _, conn := range neuron.Connections {
			if int(conn[0]) == neuronID {
				downstream = append(downstream, id)
				break
			}
		}
	}
	return downstream
}

// getWeight retrieves the weight from source to target neuron.
func (bp *Phase) getWeight(sourceID, targetID int) float64 {
	targetNeuron := bp.Neurons[targetID]
	for _, conn := range targetNeuron.Connections {
		if int(conn[0]) == sourceID {
			return conn[1]
		}
	}
	return 0
}
