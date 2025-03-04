package phase

import (
	"fmt"
	"math"
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

	// If activation not provided, pick a random one
	if activation == "" {
		activation = possibleActivations[rand.Intn(len(possibleActivations))]
	}

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

// EvaluateMetricsFromCheckpoints evaluates the model's performance using precomputed checkpoints.
// It computes three metrics:
// 1. Exact accuracy: percentage of correct predictions (in [0, 100]).
// 2. Closeness bins: distribution of how close the correct output is to 1.0 (10 bins, each in [0, 100]).
// 3. Approximate score: weighted score awarding partial credit for near-correct predictions (in [0, 100]).
func (bp *Phase) EvaluateMetricsFromCheckpoints(
	checkpoints []map[int]map[string]interface{},
	labels []float64,
) (
	exactAcc float64, // Exact accuracy in [0, 100]%
	closenessBins []float64, // Closeness bins: 0-10%, 10-20%, ..., >90%
	approxScore float64, // Approximate score in [0, 100]
) {
	nSamples := len(checkpoints)
	if nSamples == 0 || len(labels) != nSamples {
		return 0, nil, 0
	}

	numOutputs := len(bp.OutputNodes)
	if numOutputs == 0 {
		return 0, nil, 0
	}

	// Bins: ratio in [0..0.1], [0.1..0.2], …, >0.9 => 10 total
	thresholds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	binCounts := make([]float64, len(thresholds)+1)

	exactMatches := 0.0
	sumApprox := 0.0
	sampleWeight := 100.0 / float64(nSamples) // Each sample's contribution to approx score

	for i, checkpoint := range checkpoints {
		label := int(math.Round(labels[i]))
		if label < 0 || label >= numOutputs {
			continue
		}

		// Compute outputs from checkpoint
		bp.ComputeOutputsFromCheckpoint(checkpoint)

		// Gather outputs in order of OutputNodes
		vals := make([]float64, numOutputs)
		for j, id := range bp.OutputNodes {
			v := bp.Neurons[id].Value // Or use outputMap[id] if preferred
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = 0
			}
			vals[j] = v
		}

		// (1) Exact Accuracy
		predClass := argmaxFloatSlice(vals)
		if predClass == label {
			exactMatches++
		}

		// (2) Closeness Bins
		correctVal := vals[label]
		difference := math.Abs(correctVal - 1.0)
		if difference > 1 {
			difference = 1 // Clamp to [0, 1]
		}
		ratio := difference

		assigned := false
		for k, th := range thresholds {
			if ratio <= th {
				binCounts[k]++
				assigned = true
				break
			}
		}
		if !assigned {
			binCounts[len(thresholds)]++ // >90%
		}

		// (3) Approximate Score
		approx := bp.CalculatePercentageMatch(float64(label), float64(predClass))
		partialCredit := approx / 100.0
		sumApprox += partialCredit * sampleWeight
	}

	// Calculate final metrics
	exactAcc = (exactMatches / float64(nSamples)) * 100.0
	closenessBins = make([]float64, len(binCounts))
	for i := range binCounts {
		closenessBins[i] = (binCounts[i] / float64(nSamples)) * 100.0
	}
	approxScore = sumApprox

	return exactAcc, closenessBins, approxScore
}
