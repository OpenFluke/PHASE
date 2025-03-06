package phase

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

type Sample struct {
	Inputs map[int]float64
	Label  int
}

// TrainNetwork trains the network using backpropagation.
func (bp *Phase) TrainNetwork(inputs map[int]float64, expectedOutputs map[int]float64, learningRate float64, clampMin float64, clampMax float64) {
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
				if neuron.Connections[i][1] > clampMax {
					neuron.Connections[i][1] = clampMax
				} else if neuron.Connections[i][1] < clampMin {
					neuron.Connections[i][1] = clampMin
				}
			}
		}
		if !math.IsNaN(errorTerm) && !math.IsInf(errorTerm, 0) {
			neuron.Bias += learningRate * errorTerm
			// Clamp bias to [-5, 5]
			if neuron.Bias > clampMax {
				neuron.Bias = clampMax
			} else if neuron.Bias < clampMin {
				neuron.Bias = clampMin
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

func (bp *Phase) TrainNetworkTargeted(inputs map[int]float64, expectedOutputs map[int]float64, learningRate float64, clampMin float64, clampMax float64, trainableNeurons []int) {
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

	// Create a set of trainable neuron IDs for fast lookup
	trainableSet := make(map[int]struct{}, len(trainableNeurons))
	for _, id := range trainableNeurons {
		trainableSet[id] = struct{}{}
	}

	// Backward pass
	for id, neuron := range bp.Neurons {
		if neuron.Type == "input" {
			continue
		}

		// Compute error term for all neurons (needed for backpropagation)
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

		// Only update weights and bias if this neuron is in trainableNeurons
		if _, isTrainable := trainableSet[id]; isTrainable {
			// Update weights
			for i, conn := range neuron.Connections {
				sourceID := int(conn[0])
				sourceValue := bp.Neurons[sourceID].Value
				gradient := errorTerm * sourceValue
				if !math.IsNaN(gradient) && !math.IsInf(gradient, 0) {
					neuron.Connections[i][1] += learningRate * gradient
					// Clamp weight
					if neuron.Connections[i][1] > clampMax {
						neuron.Connections[i][1] = clampMax
					} else if neuron.Connections[i][1] < clampMin {
						neuron.Connections[i][1] = clampMin
					}
				}
			}

			// Update bias
			if !math.IsNaN(errorTerm) && !math.IsInf(errorTerm, 0) {
				neuron.Bias += learningRate * errorTerm
				// Clamp bias
				if neuron.Bias > clampMax {
					neuron.Bias = clampMax
				} else if neuron.Bias < clampMin {
					neuron.Bias = clampMin
				}
			}
		}
	}
}

func (bp *Phase) Grow(checkpointFolder string, originalBP *Phase, samples *[]Sample, checkpoints *[]map[int]map[string]interface{}, workerID int, maxIterations int, maxConsecutiveFailures int, minConnections int, maxConnections int, epsilon float64) ModelResult {
	bestBP := originalBP.Copy()

	var bestExactAcc float64
	var bestClosenessBins []float64
	var bestApproxScore float64

	bestExactAcc, bestClosenessBins, bestApproxScore = bestBP.EvaluateWithCheckpoints(checkpointFolder, checkpoints, GetLabels(samples))
	bestClosenessQuality := bp.ComputeClosenessQuality(bestClosenessBins)
	consecutiveFailures := 0
	iterations := 0
	neuronsAdded := 0

	for consecutiveFailures < maxConsecutiveFailures && iterations < maxIterations {

		iterations++
		currentBP := bestBP.Copy()
		numToAdd := rand.Intn(10) + 5

		for i := 0; i < numToAdd; i++ {
			newNeuron := currentBP.AddNeuronFromPreOutputs("dense", "", minConnections, maxConnections)
			if newNeuron != nil {
				currentBP.AddNewNeuronToOutput(newNeuron.ID)
				neuronsAdded++
			}
		}

		var newExactAcc float64
		var newClosenessBins []float64
		var newApproxScore float64

		newExactAcc, newClosenessBins, newApproxScore = currentBP.EvaluateWithCheckpoints(checkpointFolder, checkpoints, GetLabels(samples))
		newClosenessQuality := currentBP.ComputeClosenessQuality(newClosenessBins)
		fmt.Printf("Sandbox %d, Iter %d: eA=%.4f, cQ=%.4f, aS=%.4f, Neurons=%d\n",
			workerID, iterations, newExactAcc, newClosenessQuality, newApproxScore, neuronsAdded)

		improvedMetrics := []string{}
		if newExactAcc > bestExactAcc+epsilon {
			improvedMetrics = append(improvedMetrics, "eA")
		}
		if newClosenessQuality > bestClosenessQuality+epsilon {
			improvedMetrics = append(improvedMetrics, "cQ")
		}
		if newApproxScore > bestApproxScore+epsilon {
			improvedMetrics = append(improvedMetrics, "aS")
		}

		if len(improvedMetrics) > 0 {
			fmt.Printf("Sandbox %d: Improvement at Iter %d (%s): eA=%.4f, cQ=%.4f, aS=%.4f, Neurons=%d\n",
				workerID, iterations, strings.Join(improvedMetrics, ", "), newExactAcc, newClosenessQuality, newApproxScore, neuronsAdded)
			bestBP = currentBP
			bestExactAcc = newExactAcc
			bestClosenessBins = newClosenessBins
			bestClosenessQuality = newClosenessQuality
			bestApproxScore = newApproxScore
			consecutiveFailures = 0
		} else {
			consecutiveFailures++
		}

	}

	fmt.Printf("Sandbox %d: Exited after %d iterations, %d consecutive failures, eA=%.4f, cQ=%.4f, aS=%.4f\n",
		workerID, iterations, consecutiveFailures, bestExactAcc, bestClosenessQuality, bestApproxScore)
	return ModelResult{
		BP:            bestBP,
		ExactAcc:      bestExactAcc,
		ClosenessBins: bestClosenessBins,
		ApproxScore:   bestApproxScore,
		NeuronsAdded:  neuronsAdded,
	}
}
