package phase

import (
	"fmt"
	"math"
	"math/rand"
)

type Sample struct {
	Inputs          map[int]float64
	ExpectedOutputs map[int]float64 // Replaces Label; maps neuron IDs to target values
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

func (bp *Phase) Grow(minNeuronsToAdd int, maxNeuronsToAdd int, evalWithMultiCore bool, checkpointFolder string, originalBP *Phase, samples *[]Sample, checkpoints *[]map[int]map[string]interface{}, workerID int, maxIterations int, maxConsecutiveFailures int, minConnections int, maxConnections int, epsilon float64) ModelResult {
	bestBP := originalBP.Copy()

	var bestExactAcc float64
	var bestClosenessBins []float64
	var bestApproxScore float64

	if evalWithMultiCore {
		bestExactAcc, bestClosenessBins, bestApproxScore = bestBP.EvaluateWithCheckpointsMultiCore(checkpointFolder, checkpoints, GetLabels(samples, bestBP.OutputNodes))
	} else {
		bestExactAcc, bestClosenessBins, bestApproxScore = bestBP.EvaluateWithCheckpoints(checkpointFolder, checkpoints, GetLabels(samples, bestBP.OutputNodes))
	}
	bestClosenessQuality := bp.ComputeClosenessQuality(bestClosenessBins)

	consecutiveFailures := 0
	iterations := 0
	neuronsAdded := 0

	for consecutiveFailures < maxConsecutiveFailures && iterations < maxIterations {
		iterations++
		currentBP := bestBP.Copy()
		numToAdd := rand.Intn(maxNeuronsToAdd-minNeuronsToAdd+1) + minNeuronsToAdd

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

		if evalWithMultiCore {
			newExactAcc, newClosenessBins, newApproxScore = currentBP.EvaluateWithCheckpointsMultiCore(checkpointFolder, checkpoints, GetLabels(samples, currentBP.OutputNodes))
		} else {
			newExactAcc, newClosenessBins, newApproxScore = currentBP.EvaluateWithCheckpoints(checkpointFolder, checkpoints, GetLabels(samples, currentBP.OutputNodes))
		}
		newClosenessQuality := currentBP.ComputeClosenessQuality(newClosenessBins)

		fmt.Printf("Sandbox %d, Iter %d: eA=%.4f, cQ=%.4f, aS=%.4f, Neurons=%d\n",
			workerID, iterations, newExactAcc, newClosenessQuality, newApproxScore, neuronsAdded)

		newResult := ModelResult{
			ExactAcc:      newExactAcc,
			ClosenessBins: newClosenessBins,
			ApproxScore:   newApproxScore,
		}

		improvement := bp.ComputeTotalImprovement(newResult, bestExactAcc, bestClosenessQuality, bestApproxScore)
		if improvement > 0 {
			fmt.Printf("Sandbox %d: Improvement at Iter %d: Total Improvement=%.4f, eA=%.4f, cQ=%.4f, aS=%.4f, Neurons=%d\n",
				workerID, iterations, improvement, newExactAcc, newClosenessQuality, newApproxScore, neuronsAdded)
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

/*
func (bp *Phase) NEWGrow(minNeuronsToAdd int, maxNeuronsToAdd int, evalWithMultiCore bool, checkpointFolder string, originalBP *Phase, samples *[]Sample, checkpoints *[]map[int]map[string]interface{}, workerID int, maxIterations int, maxConsecutiveFailures int, minConnections int, maxConnections int, epsilon float64) ModelResult {
	bestBP := originalBP.Copy()
	labels := *GetLabels(samples)

	var bestExactAcc float64
	var bestClosenessBins []float64
	var bestApproxScore float64
	if evalWithMultiCore {
		bestExactAcc, bestClosenessBins, bestApproxScore = bestBP.EvaluateWithCheckpointsMultiCore(checkpointFolder, checkpoints, &labels)
	} else {
		bestExactAcc, bestClosenessBins, bestApproxScore = bestBP.EvaluateWithCheckpoints(checkpointFolder, checkpoints, &labels)
	}
	bestClosenessQuality := bp.ComputeClosenessQuality(bestClosenessBins)

	consecutiveFailures := 0
	iterations := 0
	neuronsAdded := 0

	for consecutiveFailures < maxConsecutiveFailures && iterations < maxIterations {
		iterations++
		currentBP := bestBP.Copy()
		numToAdd := rand.Intn(maxNeuronsToAdd-minNeuronsToAdd+1) + minNeuronsToAdd

		for i := 0; i < numToAdd; i++ {
			newNeuron := currentBP.AddNeuronFromPreOutputs("dense", "", minConnections, maxConnections)
			if newNeuron != nil {
				currentBP.AddNewNeuronToOutput(newNeuron.ID)
				// Optimize the new neuron
				currentBP.OptimizeNewNeuronParameters(newNeuron.ID, *checkpoints, labels, 10, 0.1, 100)
				neuronsAdded++
			}
		}

		var newExactAcc float64
		var newClosenessBins []float64
		var newApproxScore float64
		if evalWithMultiCore {
			newExactAcc, newClosenessBins, newApproxScore = currentBP.EvaluateWithCheckpointsMultiCore(checkpointFolder, checkpoints, &labels)
		} else {
			newExactAcc, newClosenessBins, newApproxScore = currentBP.EvaluateWithCheckpoints(checkpointFolder, checkpoints, &labels)
		}
		newClosenessQuality := currentBP.ComputeClosenessQuality(newClosenessBins)

		if newExactAcc > bestExactAcc+epsilon || newClosenessQuality > bestClosenessQuality+epsilon || newApproxScore > bestApproxScore+epsilon {
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

	return ModelResult{
		BP:            bestBP,
		ExactAcc:      bestExactAcc,
		ClosenessBins: bestClosenessBins,
		ApproxScore:   bestApproxScore,
		NeuronsAdded:  neuronsAdded,
	}
}*/

// OptimizeNewNeuronParameters optimizes the parameters of a newly added neuron using a perturbation-based search.
func (bp *Phase) OptimizeNewNeuronParameters(newNeuronID int, checkpoints []map[int]map[string]interface{}, labels []float64, numPerturbations int, sigma float64, maxIterations int) {
	// Get initial parameters
	currentParams := bp.GetNewNeuronParameters(newNeuronID)

	// Compute initial metrics
	currentExactAcc, currentClosenessBins, currentApproxScore := bp.EvaluateMetricsFromCheckpoints(checkpoints, labels)
	currentClosenessQuality := bp.ComputeClosenessQuality(currentClosenessBins)

	// Log initial metrics
	//fmt.Printf("Starting optimization for neuron %d. Initial Metrics: ExactAcc=%.4f, ClosenessQuality=%.4f, ApproxScore=%.4f\n",
	//	newNeuronID, currentExactAcc, currentClosenessQuality, currentApproxScore)

	for iter := 0; iter < maxIterations; iter++ {
		bestImprovement := 0.0
		bestParams := make([]float64, len(currentParams))
		copy(bestParams, currentParams)

		// Generate and evaluate perturbations
		for i := 0; i < numPerturbations; i++ {
			// Generate perturbation
			delta := make([]float64, len(currentParams))
			for j := range delta {
				delta[j] = rand.NormFloat64() * sigma
			}
			perturbedParams := make([]float64, len(currentParams))
			for j := range perturbedParams {
				perturbedParams[j] = currentParams[j] + delta[j]
			}
			bp.SetNewNeuronParameters(newNeuronID, perturbedParams)

			// Compute new metrics
			newExactAcc, newClosenessBins, newApproxScore := bp.EvaluateMetricsFromCheckpoints(checkpoints, labels)
			// := bp.ComputeClosenessQuality(newClosenessBins)

			// Compute total improvement
			improvement := bp.ComputeTotalImprovement(ModelResult{
				ExactAcc:      newExactAcc,
				ClosenessBins: newClosenessBins,
				ApproxScore:   newApproxScore,
			}, currentExactAcc, currentClosenessQuality, currentApproxScore)

			// Log perturbation metrics
			//fmt.Printf("Neuron %d, Iteration %d, Perturbation %d: Improvement=%.4f, ExactAcc=%.4f, ClosenessQuality=%.4f, ApproxScore=%.4f\n",
			//	newNeuronID, iter, i, improvement, newExactAcc, newClosenessQuality, newApproxScore)

			if improvement > bestImprovement {
				bestImprovement = improvement
				copy(bestParams, perturbedParams)
			}
		}

		// Update if improvement found
		if bestImprovement > 0 {
			currentParams = bestParams
			bp.SetNewNeuronParameters(newNeuronID, bestParams)

			// Update current metrics
			currentExactAcc, currentClosenessBins, currentApproxScore = bp.EvaluateMetricsFromCheckpoints(checkpoints, labels)
			currentClosenessQuality = bp.ComputeClosenessQuality(currentClosenessBins)

			// Log improvement
			fmt.Printf("Neuron %d, Iteration %d: Improved total improvement to %.4f\n", newNeuronID, iter, bestImprovement)
		} else {
			// Log no improvement
			fmt.Printf("Neuron %d, Iteration %d: No improvement found. Stopping optimization.\n", newNeuronID, iter)
			break
		}
	}

	// Log final metrics
	//fmt.Printf("Optimization for neuron %d completed. Final Metrics: ExactAcc=%.4f, ClosenessQuality=%.4f, ApproxScore=%.4f\n",
	//	newNeuronID, currentExactAcc, currentClosenessQuality, currentApproxScore)
}

// GetNewNeuronParameters retrieves the parameters (incoming weights, bias, outgoing weights) of a neuron.
func (bp *Phase) GetNewNeuronParameters(newNeuronID int) []float64 {
	newNeuron := bp.Neurons[newNeuronID]
	params := []float64{}
	// Incoming weights
	for _, conn := range newNeuron.Connections {
		params = append(params, conn[1])
	}
	// Bias
	params = append(params, newNeuron.Bias)
	// Outgoing weights to output neurons
	for _, outID := range bp.OutputNodes {
		outNeuron := bp.Neurons[outID]
		for _, conn := range outNeuron.Connections {
			if int(conn[0]) == newNeuronID {
				params = append(params, conn[1])
				break
			}
		}
	}
	return params
}

// SetNewNeuronParameters sets the parameters of a neuron from a slice.
func (bp *Phase) SetNewNeuronParameters(newNeuronID int, params []float64) {
	newNeuron := bp.Neurons[newNeuronID]
	idx := 0
	// Set incoming weights
	for i := range newNeuron.Connections {
		newNeuron.Connections[i][1] = params[idx]
		idx++
	}
	// Set bias
	newNeuron.Bias = params[idx]
	idx++
	// Set outgoing weights
	for _, outID := range bp.OutputNodes {
		outNeuron := bp.Neurons[outID]
		for i, conn := range outNeuron.Connections {
			if int(conn[0]) == newNeuronID {
				outNeuron.Connections[i][1] = params[idx]
				idx++
				break
			}
		}
	}
}

// EvaluateExactAccuracy computes the exact accuracy using checkpoints.
func (bp *Phase) EvaluateExactAccuracy(checkpoints []map[int]map[string]interface{}, labels []float64) float64 {
	exactAcc, _, _ := bp.EvaluateWithCheckpoints("", &checkpoints, &labels)
	return exactAcc
}

// TrainWithNeuronAdditionAndOptimization is the new training method.
func (bp *Phase) TrainWithNeuronAdditionAndOptimization(checkpoints []map[int]map[string]interface{}, labels []float64, maxNeuronsToAdd int, minConnections int, maxConnections int, numPerturbations int, sigma float64, maxIterations int) {
	initialAcc := bp.EvaluateExactAccuracy(checkpoints, labels)
	if bp.Debug {
		fmt.Printf("Initial Exact Accuracy: %.4f%%\n", initialAcc)
	}

	for i := 0; i < maxNeuronsToAdd; i++ {
		// Add a new neuron
		newNeuron := bp.AddNeuronFromPreOutputs("dense", "linear", minConnections, maxConnections)
		if newNeuron == nil {
			if bp.Debug {
				fmt.Println("Failed to add new neuron, stopping.")
			}
			break
		}
		bp.AddNewNeuronToOutput(newNeuron.ID)

		// Optimize the new neuron's parameters
		bp.OptimizeNewNeuronParameters(newNeuron.ID, checkpoints, labels, numPerturbations, sigma, maxIterations)

		// Check improvement
		newAcc := bp.EvaluateExactAccuracy(checkpoints, labels)
		if bp.Debug {
			fmt.Printf("After adding neuron %d: Exact Accuracy: %.4f%%\n", newNeuron.ID, newAcc)
		}
		if newAcc <= initialAcc {
			if bp.Debug {
				fmt.Println("No improvement detected, stopping.")
			}
			break
		}
		initialAcc = newAcc
	}

	// Apply softmax to output layer after optimization
	bp.ApplySoftmax()
	finalAcc := bp.EvaluateExactAccuracy(checkpoints, labels)
	if bp.Debug {
		fmt.Printf("Final Exact Accuracy with Softmax: %.4f%%\n", finalAcc)
	}
}
