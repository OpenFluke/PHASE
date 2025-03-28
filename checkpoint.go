package phase

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
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
func (bp *Phase) CheckpointPreOutputNeurons(checkpointFolder string, inputs []map[int]float64, timesteps int) []map[int]map[string]interface{} {
	checkpoints := make([]map[int]map[string]interface{}, len(inputs))

	for i, inputMap := range inputs {
		// Run forward pass excluding output neurons
		bp.ForwardUpTo(inputMap, timesteps, bp.OutputNodes)

		// Now compute the current set of pre-output neurons.
		preOutputIDs := bp.GetPreOutputNeurons()

		// Save their states.
		checkpoint := make(map[int]map[string]interface{})
		for _, id := range preOutputIDs {
			if neuron, exists := bp.Neurons[id]; exists {
				checkpoint[id] = bp.GetNeuronState(neuron)
			}
		}

		if checkpointFolder == "" {
			// In-memory mode: store the checkpoint in the return array
			checkpoints[i] = checkpoint
		} else {
			// File mode: save to file and leave the array entry empty
			if err := bp.SaveCheckpoint(checkpointFolder, i, checkpoint); err != nil {
				if bp.Debug {
					fmt.Printf("Checkpoint %d: Failed to save: %v\n", i, err)
				}
				// Still add an empty map to maintain array length, even on error
				checkpoints[i] = nil // or make(map[int]map[string]interface{})
				continue
			}
			checkpoints[i] = nil // or make(map[int]map[string]interface{})
		}

		if bp.Debug {
			fmt.Printf("Checkpoint %d created with %d pre-output neuron states\n", i, len(checkpoint))
		}
	}
	return checkpoints
}

// CheckpointPreOutputNeuronsMultiCore computes pre-output neuron states and optionally saves them to files using multiple cores.
// - checkpointFolder: If non-empty, saves checkpoints to files in this folder; if empty, returns them in-memory.
// - inputs: Input data to process through the network.
// - timesteps: Number of timesteps for the forward pass.
// Returns an array of checkpoints; if saved to files, the array contains nil entries matching the input length.
func (bp *Phase) CheckpointPreOutputNeuronsMultiCore(checkpointFolder string, inputs []map[int]float64, timesteps int) []map[int]map[string]interface{} {
	checkpoints := make([]map[int]map[string]interface{}, len(inputs))

	// Worker pool setup
	numWorkers := int(float64(runtime.NumCPU()) * 0.8) // Use 80% of CPU cores
	jobs := make(chan int, len(inputs))
	results := make(chan struct {
		index      int
		checkpoint map[int]map[string]interface{}
		err        error
	}, len(inputs))
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				// Each worker creates a fresh copy of the model to avoid race conditions
				localBP := bp.Copy() // Assuming Copy() creates a deep copy of the Phase struct
				inputMap := inputs[i]

				// Run forward pass excluding output neurons
				localBP.ForwardUpTo(inputMap, timesteps, localBP.OutputNodes)

				// Compute the current set of pre-output neurons
				preOutputIDs := localBP.GetPreOutputNeurons()

				// Save their states
				checkpoint := make(map[int]map[string]interface{})
				for _, id := range preOutputIDs {
					if neuron, exists := localBP.Neurons[id]; exists {
						checkpoint[id] = localBP.GetNeuronState(neuron)
					}
				}

				var err error
				if checkpointFolder != "" {
					// File mode: save to file
					err = localBP.SaveCheckpoint(checkpointFolder, i, checkpoint)
				}

				results <- struct {
					index      int
					checkpoint map[int]map[string]interface{}
					err        error
				}{i, checkpoint, err}

				if bp.Debug {
					fmt.Printf("Checkpoint %d created with %d pre-output neuron states\n", i, len(checkpoint))
				}
			}
		}()
	}

	// Send jobs
	for i := 0; i < len(inputs); i++ {
		jobs <- i
	}
	close(jobs)

	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()

	// Process results
	for res := range results {
		if checkpointFolder == "" {
			// In-memory mode: store the checkpoint
			checkpoints[res.index] = res.checkpoint
		} else {
			// File mode: check for errors and set to nil
			if res.err != nil {
				if bp.Debug {
					fmt.Printf("Checkpoint %d: Failed to save: %v\n", res.index, res.err)
				}
			}
			checkpoints[res.index] = nil // or make(map[int]map[string]interface{})
		}
	}

	return checkpoints
}

// ComputeOutputsFromCheckpoint restores the pre-output neuron states from the checkpoint
// and computes the output neurons’ values, but it filters each output neuron’s input connections
// so that only those from neurons included in the checkpoint are used.
func (bp *Phase) ComputeOutputsFromCheckpoint(checkpoint map[int]map[string]interface{}) map[int]float64 {
	// Reset non-input neurons.
	bp.ResetNeuronValues()

	// Build a set of checkpointed neuron IDs and restore their state.
	preOutputSet := make(map[int]bool)
	for id, state := range checkpoint {
		if neuron, exists := bp.Neurons[id]; exists {
			bp.SetNeuronState(neuron, state)
			preOutputSet[id] = true
		}
	}

	// Sort output nodes for consistent processing.
	sortedOutputIDs := make([]int, len(bp.OutputNodes))
	copy(sortedOutputIDs, bp.OutputNodes)
	sort.Ints(sortedOutputIDs)

	// Process each output neuron.
	for _, outID := range sortedOutputIDs {
		neuron := bp.Neurons[outID]
		inputValues := []float64{}
		for _, conn := range neuron.Connections {
			sourceID := int(conn[0])
			weight := conn[1]
			if _, ok := preOutputSet[sourceID]; ok {
				inputValues = append(inputValues, bp.Neurons[sourceID].Value*weight)
			} else if sourceNeuron, exists := bp.Neurons[sourceID]; exists {
				inputValues = append(inputValues, sourceNeuron.Value*weight)
			}
		}
		bp.ProcessNeuron(neuron, inputValues, 0)
	}

	// Collect output values.
	outputs := make(map[int]float64)
	for _, id := range bp.OutputNodes {
		outputs[id] = bp.Neurons[id].Value
	}
	return outputs
}

// AddNeuronFromPreOutputs creates a new neuron whose incoming connections
// are taken from a random subset of the pre-output neurons, then adds a connection
// from the new neuron to every output neuron (without removing existing connections).
func (bp *Phase) AddNeuronFromPreOutputs(neuronType, activation string, minConnections, maxConnections int) *Neuron {
	// If no activation is provided, choose one randomly from a predefined list.
	if activation == "" {
		activation = possibleActivations[rand.Intn(len(possibleActivations))]
	}

	// If no neuron type is provided, choose one randomly from a predefined list.
	if neuronType == "" {
		neuronType = neuronTypes[rand.Intn(len(neuronTypes))]
	}

	// Get the IDs of neurons that feed into the output layer.
	preOutputIDs := bp.GetPreOutputNeurons()
	if len(preOutputIDs) == 0 {
		return nil // Cannot add a neuron if there are no pre-output neurons.
	}

	// Determine the number of incoming connections, constrained by available neurons.
	numConns := rand.Intn(maxConnections-minConnections+1) + minConnections
	if numConns > len(preOutputIDs) {
		numConns = len(preOutputIDs)
	}

	// Shuffle the pre-output neuron IDs and select a subset for connections.
	rand.Shuffle(len(preOutputIDs), func(i, j int) {
		preOutputIDs[i], preOutputIDs[j] = preOutputIDs[j], preOutputIDs[i]
	})
	selectedIDs := preOutputIDs[:numConns]

	// Create the new neuron with basic properties.
	newID := bp.GetNextNeuronID()
	newNeuron := &Neuron{
		ID:          newID,
		Type:        neuronType,
		Bias:        rand.NormFloat64() * 0.1, // Small random bias
		Activation:  activation,
		Connections: make([][]float64, 0, numConns),
		IsNew:       true, // Mark as newly added
	}

	// Add incoming connections from the selected pre-output neurons with small random weights.
	for _, srcID := range selectedIDs {
		weight := rand.NormFloat64() * 0.1
		newNeuron.Connections = append(newNeuron.Connections, []float64{float64(srcID), weight})
	}

	// Initialize type-specific parameters based on neuronType.
	switch neuronType {
	case "lstm":
		// Initialize gate weights for LSTM neurons based on the number of connections.
		conCount := len(newNeuron.Connections)
		newNeuron.GateWeights = map[string][]float64{
			"input":  bp.RandomWeights(conCount), // Random weights for input gate
			"forget": bp.RandomWeights(conCount), // Random weights for forget gate
			"output": bp.RandomWeights(conCount), // Random weights for output gate
			"cell":   bp.RandomWeights(conCount), // Random weights for cell gate
		}
	case "cnn":
		// Randomize the number of kernels between 1 and 10 for CNN neurons.
		numKernels := rand.Intn(10) + 1 // Generates a random integer from 1 to 10
		newNeuron.Kernels = make([][]float64, numKernels)
		for i := 0; i < numKernels; i++ {
			// Initialize each kernel as a 2x2 matrix with random values.
			kernel := make([]float64, 4) // 2x2 = 4 elements
			for j := 0; j < 4; j++ {
				kernel[j] = rand.Float64() // Random float between 0 and 1
			}
			newNeuron.Kernels[i] = kernel
		}
	case "batch_norm":
		// Initialize batch normalization parameters.
		newNeuron.BatchNormParams = &BatchNormParams{
			Gamma: 1.0, // Scaling factor
			Beta:  0.0, // Shift factor
			Mean:  0.0, // Running mean
			Var:   1.0, // Running variance
		}
	default:
		// For "dense" or unrecognized types, no additional initialization is required.
	}

	// Add the new neuron to the network's neuron map.
	bp.Neurons[newID] = newNeuron

	// Connect the new neuron to every output neuron.
	bp.AddNewNeuronToOutput(newID)

	return newNeuron
}

// AddNewNeuronToOutput connects the new neuron to every output neuron by adding
// a new connection with a small random weight if one does not already exist.
func (bp *Phase) AddNewNeuronToOutput(newNeuronID int) {
	for _, outID := range bp.OutputNodes {
		outNeuron := bp.Neurons[outID]
		if !bp.connectionExists(newNeuronID, outID) {
			weight := rand.NormFloat64() * 0.1 // small random weight
			outNeuron.Connections = append(outNeuron.Connections, []float64{float64(newNeuronID), weight})
			if bp.Debug {
				fmt.Printf("Added connection from new neuron %d to output neuron %d with weight %f\n", newNeuronID, outID, weight)
			}
		}
	}
}

// EvaluateMetricsFromCheckpoints evaluates the model's performance using precomputed checkpoints.
// It computes three metrics:
// 1. Exact accuracy: percentage of correct predictions (in [0, 100]).
// 2. Closeness bins: distribution of how close the correct output is to 1.0 (10 bins, each in [0, 100]).
// 3. Approximate score: weighted score awarding partial credit for near-correct predictions (in [0, 100]).
// EvaluateMetricsFromCheckpoints evaluates the model's performance using precomputed checkpoints.
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

	thresholds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	binCounts := make([]float64, len(thresholds)+1)

	exactMatches := 0.0
	sumApprox := 0.0
	sampleWeight := 100.0 / float64(nSamples)

	for i, checkpoint := range checkpoints {
		label := int(math.Round(labels[i]))
		if label < 0 || label >= numOutputs {
			continue
		}

		// Use the new function to compute outputs, including new neurons
		bp.ComputeOutputsWithNewNeuronsFromCheckpoint(checkpoint)

		vals := make([]float64, numOutputs)
		for j, id := range bp.OutputNodes {
			v := bp.Neurons[id].Value
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = 0
			}
			vals[j] = v
		}

		// Exact Accuracy
		predClass := argmaxFloatSlice(vals)
		if predClass == label {
			exactMatches++
		}

		// Closeness Bins
		correctVal := vals[label]
		difference := math.Abs(correctVal - 1.0)
		if difference > 1 {
			difference = 1
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
			binCounts[len(thresholds)]++
		}

		// Approximate Score
		approx := bp.CalculatePercentageMatch(float64(label), float64(predClass))
		partialCredit := approx / 100.0
		sumApprox += partialCredit * sampleWeight
	}

	exactAcc = (exactMatches / float64(nSamples)) * 100.0
	closenessBins = make([]float64, len(binCounts))
	for i := range binCounts {
		closenessBins[i] = (binCounts[i] / float64(nSamples)) * 100.0
	}
	approxScore = sumApprox

	return exactAcc, closenessBins, approxScore
}

// CheckpointAllHiddenNeurons runs a forward pass and saves the state of all non-input neurons.
func (bp *Phase) CheckpointAllHiddenNeurons(inputs []map[int]float64, timesteps int) []map[int]map[string]interface{} {
	checkpoints := make([]map[int]map[string]interface{}, len(inputs))
	for i, inputMap := range inputs {
		// Run the full forward pass (including all hidden neurons)
		bp.Forward(inputMap, timesteps)
		checkpoint := make(map[int]map[string]interface{})
		// Save state for every neuron that is not an input.
		for id, neuron := range bp.Neurons {
			if neuron.Type != "input" {
				checkpoint[id] = bp.GetNeuronState(neuron)
			}
		}
		checkpoints[i] = checkpoint
		if bp.Debug {
			fmt.Printf("Checkpoint %d created with %d hidden neuron states\n", i, len(checkpoint))
		}
	}
	return checkpoints
}

// ComputeOutputsFromFullCheckpoint restores the state of all hidden neurons from the checkpoint
// and then returns the output neuron values.
func (bp *Phase) ComputeOutputsFromFullCheckpoint(checkpoint map[int]map[string]interface{}) map[int]float64 {
	// Reset non-input neurons.
	bp.ResetNeuronValues()

	// Restore state for every non-input neuron from the checkpoint.
	for id, state := range checkpoint {
		if neuron, exists := bp.Neurons[id]; exists {
			bp.SetNeuronState(neuron, state)
		}
	}

	// (Optionally) you might run a final propagation step if needed.
	// For many feedforward networks, the output neurons' values are already restored.

	// Collect output values.
	outputs := make(map[int]float64)
	for _, id := range bp.OutputNodes {
		outputs[id] = bp.Neurons[id].Value
	}
	return outputs
}

// ComputeOutputsWithNewNeurons restores the checkpointed state and computes outputs with sample inputs
func (bp *Phase) ComputeOutputsWithNewNeurons(checkpoint map[int]map[string]interface{}, inputs map[int]float64, timesteps int) map[int]float64 {
	bp.ResetNeuronValues()

	// Set sample-specific inputs
	for id, value := range inputs {
		if neuron, exists := bp.Neurons[id]; exists {
			neuron.Value = value
			if bp.Debug {
				fmt.Printf("Input Neuron %d set to %f\n", id, value)
			}
		}
	}

	// Restore checkpointed state for non-input neurons
	for id, state := range checkpoint {
		if neuron, exists := bp.Neurons[id]; exists {
			bp.SetNeuronState(neuron, state)
			if bp.Debug {
				fmt.Printf("Restored Neuron %d: Value=%f\n", id, neuron.Value)
			}
		}
	}

	// Check if there are any new neurons
	hasNewNeurons := false
	for _, neuron := range bp.Neurons {
		if neuron.IsNew {
			hasNewNeurons = true
			break
		}
	}

	// Process neurons over timesteps
	for t := 0; t < timesteps; t++ {
		if bp.Debug {
			fmt.Printf("=== Processing Timestep %d ===\n", t)
		}

		if hasNewNeurons {
			// First pass: only new neurons
			for id := 1; id <= len(bp.Neurons); id++ {
				neuron, exists := bp.Neurons[id]
				if !exists || !neuron.IsNew {
					continue
				}
				inputValues := bp.gatherInputs(neuron)
				bp.ProcessNeuron(neuron, inputValues, t)
				if bp.Debug {
					fmt.Printf("Dense Neuron %d: Value=%f\n", id, neuron.Value)
				}
			}
		}

		// Second pass: output neurons
		for _, id := range bp.OutputNodes {
			neuron := bp.Neurons[id]
			inputValues := bp.gatherInputs(neuron)
			bp.ProcessNeuron(neuron, inputValues, t)
			if bp.Debug {
				fmt.Printf("Dense Neuron %d: Value=%f\n", id, neuron.Value)
			}
		}
	}

	return bp.GetOutputs()
}

// gatherInputs collects the weighted inputs from a neuron's upstream connections.
// It returns a slice of float64 values representing the weighted contributions from source neurons.
func (bp *Phase) gatherInputs(neuron *Neuron) []float64 {
	inputValues := make([]float64, 0, len(neuron.Connections))
	for _, conn := range neuron.Connections {
		sourceID := int(conn[0]) // The ID of the source neuron
		weight := conn[1]        // The connection weight
		if sourceNeuron, exists := bp.Neurons[sourceID]; exists {
			inputValues = append(inputValues, sourceNeuron.Value*weight)
		} else {
			// Handle missing source neuron (e.g., log a warning if debugging)
			if bp.Debug {
				fmt.Printf("Warning: Source neuron %d not found for neuron %d\n", sourceID, neuron.ID)
			}
			inputValues = append(inputValues, 0.0) // Default to zero contribution
		}
	}
	return inputValues
}

// ComputeOutputsWithNewNeuronsFromCheckpoint computes outputs from a checkpoint, including contributions from new neurons.
func (bp *Phase) ComputeOutputsWithNewNeuronsFromCheckpoint(checkpoint map[int]map[string]interface{}) map[int]float64 {
	// Reset all neuron values to zero
	bp.ResetNeuronValues()

	// Restore states of neurons in the checkpoint
	for id, state := range checkpoint {
		if neuron, exists := bp.Neurons[id]; exists {
			bp.SetNeuronState(neuron, state)
			if bp.Debug {
				fmt.Printf("Restored Neuron %d: Value=%f\n", id, neuron.Value)
			}
		}
	}

	// Process neurons not in the checkpoint (new neurons), excluding inputs
	for id, neuron := range bp.Neurons {
		if _, inCheckpoint := checkpoint[id]; !inCheckpoint && neuron.Type != "input" {
			inputValues := bp.gatherInputs(neuron)
			bp.ProcessNeuron(neuron, inputValues, 0)
			if bp.Debug {
				fmt.Printf("Processed new Neuron %d: Value=%f\n", id, neuron.Value)
			}
		}
	}

	// Process output neurons, incorporating all connections (old and new)
	for _, id := range bp.OutputNodes {
		neuron := bp.Neurons[id]
		inputValues := bp.gatherInputs(neuron)
		bp.ProcessNeuron(neuron, inputValues, 0)
		if bp.Debug {
			fmt.Printf("Processed output Neuron %d: Value=%f\n", id, neuron.Value)
		}
	}

	// Return the output values
	return bp.GetOutputs()
}

// ComputePartialOutputsFromCheckpoint restores the state of neurons that were checkpointed,
// then computes the outputs only for the new neurons (those not in the checkpoint),
// and finally processes the output neurons.
func (bp *Phase) ComputePartialOutputsFromCheckpoint(checkpoint map[int]map[string]interface{}) map[int]float64 {
	// Restore checkpointed neurons.
	for id, state := range checkpoint {
		if neuron, exists := bp.Neurons[id]; exists {
			bp.SetNeuronState(neuron, state)
		}
	}
	// Process neurons that are not in the checkpoint (new neurons).
	for id, neuron := range bp.Neurons {
		if neuron.Type == "input" {
			continue
		}
		if _, exists := checkpoint[id]; !exists {
			inputValues := bp.gatherInputs(neuron)
			bp.ProcessNeuron(neuron, inputValues, 0)
		}
	}
	// Update output neurons.
	for _, outID := range bp.OutputNodes {
		neuron := bp.Neurons[outID]
		inputValues := bp.gatherInputs(neuron)
		bp.ProcessNeuron(neuron, inputValues, 0)
	}
	// Apply softmax if the output neurons use "softmax" activation.
	if len(bp.OutputNodes) > 0 && bp.Neurons[bp.OutputNodes[0]].Activation == "softmax" {
		bp.ApplySoftmax()
	}
	return bp.GetOutputs()
}

// EvaluateWithCheckpoints evaluates the model's performance using precomputed pre-output checkpoints.
// It computes three metrics:
// 1. Exact accuracy: percentage of correct predictions (in [0, 100]).
// 2. Closeness bins: distribution of how close the correct output is to 1.0 (10 bins, each in [0, 100]).
// 3. Approximate score: weighted score awarding partial credit for near-correct predictions (in [0, 100]).
func (bp *Phase) EvaluateWithCheckpoints(checkpointFolder string, checkpoints *[]map[int]map[string]interface{}, labels *[]float64) (exactAcc float64, closenessBins []float64, approxScore float64) {
	nSamples := len(*checkpoints) // Dereference to get the length
	if nSamples == 0 || len(*labels) != nSamples {
		return 0, nil, 0
	}

	numOutputs := len(bp.OutputNodes)
	if numOutputs == 0 {
		return 0, nil, 0
	}

	// Initialize metrics variables
	thresholds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	binCounts := make([]float64, len(thresholds)+1)
	exactMatches := 0.0
	sumApprox := 0.0
	sampleWeight := 100.0 / float64(nSamples)

	// Process each sample using the checkpoint
	for i, checkpoint := range *checkpoints { // Dereference checkpoints
		label := int((*labels)[i]) // Dereference labels and access the i-th element
		if label < 0 || label >= numOutputs {
			if bp.Debug {
				fmt.Printf("Sample %d: Invalid label %d (out of range 0-%d), skipping\n", i, label, numOutputs-1)
			}
			continue
		}

		// Compute outputs using the pre-output checkpoint
		var outputs map[int]float64
		if checkpointFolder == "" {
			outputs = bp.ComputePartialOutputsFromCheckpoint(checkpoint)
		} else {
			checkpoint, err := bp.LoadCheckpoint(checkpointFolder, i)
			if err != nil {
				if bp.Debug {
					fmt.Printf("Sample %d: Failed to load checkpoint: %v, skipping\n", i, err)
				}
				continue
			}
			outputs = bp.ComputePartialOutputsFromCheckpoint(checkpoint)
		}

		// Convert outputs map to slice aligned with OutputNodes
		vals := make([]float64, numOutputs)
		for j, outID := range bp.OutputNodes {
			v := outputs[outID]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = 0
				if bp.Debug {
					fmt.Printf("Sample %d: Output neuron %d value is NaN/Inf, set to 0\n", i, outID)
				}
			}
			vals[j] = v
		}

		// Exact Accuracy: Check if argmax matches label
		predClass := argmaxFloatSlice(vals)
		if predClass == label {
			exactMatches++
		}

		// Closeness Bins: Measure how close the correct output is to 1.0
		correctVal := vals[label]
		difference := math.Abs(correctVal - 1.0)
		if difference > 1 {
			difference = 1 // Clamp difference to [0, 1]
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
			binCounts[len(thresholds)]++ // >90% bin
		}

		// Approximate Score: Award partial credit
		approx := bp.CalculatePercentageMatch(float64(label), float64(predClass))
		partialCredit := approx / 100.0
		sumApprox += partialCredit * sampleWeight

		if bp.Debug {
			fmt.Printf("Sample %d: Label=%d, Pred=%d, CorrectVal=%.4f, Outputs=%v\n", i, label, predClass, correctVal, vals)
		}
	}

	// Compute final metrics
	exactAcc = (exactMatches / float64(nSamples)) * 100.0
	closenessBins = make([]float64, len(binCounts))
	for i := range binCounts {
		closenessBins[i] = (binCounts[i] / float64(nSamples)) * 100.0
	}
	approxScore = sumApprox

	if bp.Debug {
		fmt.Printf("Evaluation complete: ExactAcc=%.2f%%, ClosenessBins=%v, ApproxScore=%.2f\n", exactAcc, closenessBins, approxScore)
	}

	return exactAcc, closenessBins, approxScore
}

func (bp *Phase) EvaluateWithCheckpointsMultiCore(checkpointFolder string, checkpoints *[]map[int]map[string]interface{}, labels *[]float64) (exactAcc float64, closenessBins []float64, approxScore float64) {
	nSamples := len(*checkpoints)
	if nSamples == 0 || len(*labels) != nSamples {
		return 0, nil, 0
	}

	numOutputs := len(bp.OutputNodes)
	if numOutputs == 0 {
		return 0, nil, 0
	}

	// Initialize metrics variables
	thresholds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	binCounts := make([]float64, len(thresholds)+1)
	exactMatches := float64(0)
	sumApprox := float64(0)
	sampleWeight := 100.0 / float64(nSamples)

	// Worker pool setup
	numWorkers := int(float64(runtime.NumCPU()) * 0.8)
	jobs := make(chan int, nSamples)
	results := make(chan struct {
		exactMatch   float64
		binIndex     int
		approxCredit float64
		err          error
	}, nSamples)
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				label := int((*labels)[i])
				if label < 0 || label >= numOutputs {
					if bp.Debug {
						fmt.Printf("Sample %d: Invalid label %d (out of range 0-%d), skipping\n", i, label, numOutputs-1)
					}
					results <- struct {
						exactMatch   float64
						binIndex     int
						approxCredit float64
						err          error
					}{0, -1, 0, nil}
					continue
				}

				var outputs map[int]float64
				if checkpointFolder == "" {
					outputs = bp.ComputePartialOutputsFromCheckpoint((*checkpoints)[i])
				} else {
					checkpoint, err := bp.LoadCheckpoint(checkpointFolder, i)
					if err != nil {
						if bp.Debug {
							fmt.Printf("Sample %d: Failed to load checkpoint: %v, skipping\n", i, err)
						}
						results <- struct {
							exactMatch   float64
							binIndex     int
							approxCredit float64
							err          error
						}{0, -1, 0, err}
						continue
					}
					outputs = bp.ComputePartialOutputsFromCheckpoint(checkpoint)
				}

				vals := make([]float64, numOutputs)
				for j, outID := range bp.OutputNodes {
					v := outputs[outID]
					if math.IsNaN(v) || math.IsInf(v, 0) {
						v = 0
						if bp.Debug {
							fmt.Printf("Sample %d: Output neuron %d value is NaN/Inf, set to 0\n", i, outID)
						}
					}
					vals[j] = v
				}

				predClass := argmaxFloatSlice(vals)
				exactMatch := 0.0
				if predClass == label {
					exactMatch = 1.0
				}

				correctVal := vals[label]
				difference := math.Abs(correctVal - 1.0)
				if difference > 1 {
					difference = 1
				}
				ratio := difference
				binIndex := len(thresholds) // Default to >90% bin
				for k, th := range thresholds {
					if ratio <= th {
						binIndex = k
						break
					}
				}

				approx := bp.CalculatePercentageMatch(float64(label), float64(predClass))
				partialCredit := approx / 100.0

				if bp.Debug {
					fmt.Printf("Sample %d: Label=%d, Pred=%d, CorrectVal=%.4f, Outputs=%v\n", i, label, predClass, correctVal, vals)
				}

				results <- struct {
					exactMatch   float64
					binIndex     int
					approxCredit float64
					err          error
				}{exactMatch, binIndex, partialCredit * sampleWeight, nil}
			}
		}()
	}

	// Send jobs
	for i := 0; i < nSamples; i++ {
		jobs <- i
	}
	close(jobs)

	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()

	var mu sync.Mutex
	for res := range results {
		if res.err != nil {
			continue // Skip failed samples
		}
		mu.Lock()
		exactMatches += res.exactMatch
		if res.binIndex >= 0 {
			binCounts[res.binIndex]++
		}
		sumApprox += res.approxCredit
		mu.Unlock()
	}

	// Compute final metrics
	exactAcc = (exactMatches / float64(nSamples)) * 100.0
	closenessBins = make([]float64, len(binCounts))
	for i := range binCounts {
		closenessBins[i] = (binCounts[i] / float64(nSamples)) * 100.0
	}
	approxScore = sumApprox

	if bp.Debug {
		fmt.Printf("Evaluation complete: ExactAcc=%.2f%%, ClosenessBins=%v, ApproxScore=%.2f\n", exactAcc, closenessBins, approxScore)
	}

	return exactAcc, closenessBins, approxScore
}
