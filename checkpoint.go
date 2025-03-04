package phase

import (
	"fmt"
	"math/rand"
	"sort"
)

// GetNeuronState captures the dynamic state of a neuron as a map.
// This allows flexibility for different neuron types (e.g., LSTM).
func (bp *Phase) GetNeuronState(neuron *Neuron) map[string]interface{} {
	state := make(map[string]interface{})
	state["Value"] = neuron.Value
	if neuron.Type == "lstm" {
		state["CellState"] = neuron.CellState
	}
	// Add support for additional neuron types here as needed.
	return state
}

// SetNeuronState restores the dynamic state of a neuron from a map.
// It matches the state variables to the neuron's type.
func (bp *Phase) SetNeuronState(neuron *Neuron, state map[string]interface{}) {
	if val, ok := state["Value"]; ok {
		neuron.Value = val.(float64)
	}
	if neuron.Type == "lstm" {
		if val, ok := state["CellState"]; ok {
			neuron.CellState = val.(float64)
		}
	}
	// Add support for additional neuron types here as needed.
}

// GetPreOutputNeurons identifies neurons directly connected to output neurons.
// These are typically the last hidden layer neurons in a feedforward network.
func (bp *Phase) GetPreOutputNeurons() []int {
	sourceSet := make(map[int]struct{})
	for _, outputID := range bp.OutputNodes {
		outputNeuron := bp.Neurons[outputID]
		for _, conn := range outputNeuron.Connections {
			sourceID := int(conn[0])
			sourceSet[sourceID] = struct{}{}
		}
	}
	sourceIDs := []int{}
	for id := range sourceSet {
		sourceIDs = append(sourceIDs, id)
	}
	return sourceIDs
}

// ResetNeuronValues resets all non-input neuron states to zero.
// This ensures no residual values affect subsequent computations.
func (bp *Phase) ResetNeuronValues() {
	for _, neuron := range bp.Neurons {
		if neuron.Type != "input" {
			neuron.Value = 0.0
			// Reset additional state variables for specific neuron types.
			if neuron.Type == "lstm" {
				neuron.CellState = 0.0
			}
		}
	}
}

// ForwardUpTo performs a forward pass excluding specified neurons (e.g., output neurons).
// It processes the network up to the pre-output neurons over multiple timesteps if needed.
func (bp *Phase) ForwardUpTo(inputs map[int]float64, timesteps int, exclude []int) {
	bp.ResetNeuronValues() // Start with a clean state

	// Set input neuron values
	for id, value := range inputs {
		if neuron, exists := bp.Neurons[id]; exists {
			neuron.Value = value
			if bp.Debug {
				fmt.Printf("Input Neuron %d set to %f\n", id, value)
			}
		}
	}

	// Create a set of excluded neuron IDs for efficient lookup
	excludeSet := make(map[int]struct{})
	for _, id := range exclude {
		excludeSet[id] = struct{}{}
	}

	// Process neurons over timesteps, skipping excluded and input neurons
	for t := 0; t < timesteps; t++ {
		if bp.Debug {
			fmt.Printf("=== Timestep %d ===\n", t)
		}
		for id := 1; id <= len(bp.Neurons); id++ {
			if _, excluded := excludeSet[id]; excluded {
				continue
			}
			neuron, exists := bp.Neurons[id]
			if !exists || neuron.Type == "input" {
				continue
			}
			inputValues := []float64{}
			for _, conn := range neuron.Connections {
				sourceID := int(conn[0])
				weight := conn[1]
				if sourceNeuron, exists := bp.Neurons[sourceID]; exists {
					inputValues = append(inputValues, sourceNeuron.Value*weight)
				}
			}
			bp.ProcessNeuron(neuron, inputValues, t)
			if bp.Debug {
				fmt.Printf("Neuron %d computed: Value=%f\n", id, neuron.Value)
			}
		}
	}
}

// CheckpointPreOutputNeurons computes up to pre-output neurons and saves their states.
// It processes a batch of inputs and returns a slice of checkpoints.
func (bp *Phase) CheckpointPreOutputNeurons(inputs []map[int]float64, timesteps int) []map[int]map[string]interface{} {
	preOutputIDs := bp.GetPreOutputNeurons()
	checkpoints := make([]map[int]map[string]interface{}, len(inputs))

	for i, inputMap := range inputs {
		// Run forward pass excluding output neurons
		bp.ForwardUpTo(inputMap, timesteps, bp.OutputNodes)

		// Save pre-output neuron states
		checkpoint := make(map[int]map[string]interface{})
		for _, id := range preOutputIDs {
			if neuron, exists := bp.Neurons[id]; exists {
				checkpoint[id] = bp.GetNeuronState(neuron)
			}
		}
		checkpoints[i] = checkpoint

		if bp.Debug {
			fmt.Printf("Checkpoint %d created with %d pre-output neuron states\n", i, len(checkpoint))
		}
	}
	return checkpoints
}

// ComputeOutputsFromCheckpoint restores pre-output states and computes output values.
// It ensures a clean state before computation to avoid interference.
func (bp *Phase) ComputeOutputsFromCheckpoint(checkpoint map[int]map[string]interface{}) map[int]float64 {
	bp.ResetNeuronValues() // Reset all non-input neurons

	// Restore pre-output neuron states from the checkpoint
	for id, state := range checkpoint {
		if neuron, exists := bp.Neurons[id]; exists {
			bp.SetNeuronState(neuron, state)
		}
	}

	// Sort output nodes for consistent processing order
	sortedOutputIDs := make([]int, len(bp.OutputNodes))
	copy(sortedOutputIDs, bp.OutputNodes)
	sort.Ints(sortedOutputIDs)

	// Compute each output neuron
	for _, id := range sortedOutputIDs {
		neuron := bp.Neurons[id]
		inputValues := []float64{}
		for _, conn := range neuron.Connections {
			sourceID := int(conn[0])
			weight := conn[1]
			if sourceNeuron, exists := bp.Neurons[sourceID]; exists {
				inputValues = append(inputValues, sourceNeuron.Value*weight)
			}
		}
		bp.ProcessNeuron(neuron, inputValues, 0) // Single timestep computation
		if bp.Debug {
			fmt.Printf("Output Neuron %d computed: Value=%f\n", id, neuron.Value)
		}
	}

	// Collect output values
	outputs := make(map[int]float64)
	for _, id := range bp.OutputNodes {
		outputs[id] = bp.Neurons[id].Value
	}
	return outputs
}

// AddNeuronFromPreOutputs adds a new neuron connected from a random subset of pre-output neurons
// and wired to all output neurons.
func (bp *Phase) AddNeuronFromPreOutputs(neuronType, activation string, minConnections, maxConnections int) *Neuron {
	// Get the IDs of pre-output neurons (neurons connected to outputs)
	preOutputIDs := bp.GetPreOutputNeurons()
	if len(preOutputIDs) == 0 {
		return nil
	}

	// Determine number of incoming connections
	numConns := rand.Intn(maxConnections-minConnections+1) + minConnections
	if numConns > len(preOutputIDs) {
		numConns = len(preOutputIDs)
	}

	// Randomly select pre-output neurons
	selectedIDs := make([]int, len(preOutputIDs))
	copy(selectedIDs, preOutputIDs)
	rand.Shuffle(len(selectedIDs), func(i, j int) {
		selectedIDs[i], selectedIDs[j] = selectedIDs[j], selectedIDs[i]
	})
	selectedIDs = selectedIDs[:numConns]

	// Create the new neuron
	newID := bp.GetNextNeuronID()
	newNeuron := &Neuron{
		ID:          newID,
		Type:        neuronType,
		Bias:        rand.NormFloat64() * 0.1,
		Activation:  activation,
		Connections: make([][]float64, 0, numConns),
	}

	// Add incoming connections from selected pre-output neurons
	for _, sourceID := range selectedIDs {
		weight := rand.NormFloat64() * 0.1
		newNeuron.Connections = append(newNeuron.Connections, []float64{float64(sourceID), weight})
	}

	// Add the neuron to the network
	bp.Neurons[newID] = newNeuron

	// Connect the new neuron to all output neurons
	bp.RewireOutputsThroughNewNeuron(newID)

	return newNeuron
}
