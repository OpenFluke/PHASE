package phase

import "math/rand"

// crossoverphases merges two parent phases to create an offspring phase.
func crossoverphases(parentA, parentB *phase) *phase {
	offspring := Newphase()

	// 1. Merge Neurons (Random Selection from Both Parents)
	for id, neuronA := range parentA.Neurons {
		if neuronB, exists := parentB.Neurons[id]; exists {
			offspring.Neurons[id] = deepCopyNeuron(selectNeuron(neuronA, neuronB))
		} else {
			offspring.Neurons[id] = deepCopyNeuron(neuronA)
		}
	}

	for id, neuronB := range parentB.Neurons {
		if _, exists := offspring.Neurons[id]; !exists {
			offspring.Neurons[id] = deepCopyNeuron(neuronB)
		}
	}

	// 2. Merge Connections (Randomly Inherit or Average)
	mergeNeuronConnections(offspring, parentA, parentB)

	// 3. Apply Activation Crossover (Random Selection)
	applyActivationCrossover(offspring, parentA, parentB)

	// 4. Ensure All Output Neurons Exist
	ensureOutputNeurons(offspring, parentA.OutputNodes)

	return offspring
}

// selectNeuron randomly chooses a neuron from either of the parents.
func selectNeuron(neuronA, neuronB *Neuron) *Neuron {
	if rand.Float64() < 0.5 {
		return neuronA
	}
	return neuronB
}

// mergeNeuronConnections merges neuron connections from both parents.
func mergeNeuronConnections(offspring, parentA, parentB *phase) {
	for id, neuron := range offspring.Neurons {
		if parentA.Neurons[id] != nil && parentB.Neurons[id] != nil {
			for i := range neuron.Connections {
				if rand.Float64() < 0.5 {
					neuron.Connections[i][1] = parentA.Neurons[id].Connections[i][1]
				} else {
					neuron.Connections[i][1] = parentB.Neurons[id].Connections[i][1]
				}
			}
		}
	}
}

// applyActivationCrossover selects the activation function from either parent.
func applyActivationCrossover(offspring, parentA, parentB *phase) {
	for id, neuron := range offspring.Neurons {
		if parentA.Neurons[id] != nil && parentB.Neurons[id] != nil {
			neuron.Activation = selectActivation(parentA.Neurons[id].Activation, parentB.Neurons[id].Activation)
		}
	}
}

// selectActivation randomly chooses an activation function from either parent.
func selectActivation(activationA, activationB string) string {
	if rand.Float64() < 0.5 {
		return activationA
	}
	return activationB
}

// ensureOutputNeurons ensures that all output neurons exist in the offspring phase.
func ensureOutputNeurons(offspring *phase, outputNodes []int) {
	for _, outputID := range outputNodes {
		if _, exists := offspring.Neurons[outputID]; !exists {
			offspring.Neurons[outputID] = &Neuron{
				ID:         outputID,
				Type:       "dense",
				Activation: "linear",
				Bias:       0,
			}
		}
	}
}

// deepCopyNeuron creates an independent deep copy of a neuron.
func deepCopyNeuron(n *Neuron) *Neuron {
	if n == nil {
		return nil
	}

	newNeuron := &Neuron{
		ID:          n.ID,
		Type:        n.Type,
		Value:       n.Value,
		Bias:        n.Bias,
		Activation:  n.Activation,
		LoopCount:   n.LoopCount,
		WindowSize:  n.WindowSize,
		DropoutRate: n.DropoutRate,
		BatchNorm:   n.BatchNorm,
		Attention:   n.Attention,
		CellState:   n.CellState,
	}

	// Deep copy arrays and maps
	copyFloat64Slice(&newNeuron.AttentionWeights, n.AttentionWeights)
	copyNeuronConnections(&newNeuron.Connections, n.Connections)
	copyLSTMGateWeights(&newNeuron.GateWeights, n.GateWeights)
	copyKernels(&newNeuron.Kernels, n.Kernels)
	copyBatchNormParams(&newNeuron.BatchNormParams, n.BatchNormParams)
	copyIntSlice(&newNeuron.NeighborhoodIDs, n.NeighborhoodIDs)
	copyFloat64Slice(&newNeuron.NCAState, n.NCAState)

	return newNeuron
}

// copyFloat64Slice safely copies a slice of float64 values.
func copyFloat64Slice(dst *[]float64, src []float64) {
	if src != nil {
		*dst = make([]float64, len(src))
		copy(*dst, src)
	}
}

// copyNeuronConnections safely copies neuron connection weights.
func copyNeuronConnections(dst *[][]float64, src [][]float64) {
	if len(src) > 0 {
		*dst = make([][]float64, len(src))
		for i, conn := range src {
			(*dst)[i] = make([]float64, len(conn))
			copy((*dst)[i], conn)
		}
	}
}

// copyLSTMGateWeights safely copies LSTM gate weights.
func copyLSTMGateWeights(dst *map[string][]float64, src map[string][]float64) {
	if src != nil {
		*dst = make(map[string][]float64)
		for key, weights := range src {
			copiedWeights := make([]float64, len(weights))
			copy(copiedWeights, weights)
			(*dst)[key] = copiedWeights
		}
	}
}

// copyKernels safely copies CNN kernels.
func copyKernels(dst *[][]float64, src [][]float64) {
	if len(src) > 0 {
		*dst = make([][]float64, len(src))
		for i, kernel := range src {
			(*dst)[i] = make([]float64, len(kernel))
			copy((*dst)[i], kernel)
		}
	}
}

// copyBatchNormParams safely copies batch normalization parameters.
func copyBatchNormParams(dst **BatchNormParams, src *BatchNormParams) {
	if src != nil {
		*dst = &BatchNormParams{
			Gamma: src.Gamma,
			Beta:  src.Beta,
			Mean:  src.Mean,
			Var:   src.Var,
		}
	}
}

// copyIntSlice safely copies a slice of int values.
func copyIntSlice(dst *[]int, src []int) {
	if src != nil {
		*dst = make([]int, len(src))
		copy(*dst, src)
	}
}
