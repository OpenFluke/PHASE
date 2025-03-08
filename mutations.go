package phase

import (
	"fmt"
	"math/rand"
)

// Possible neuron types for mutation
var neuronTypes = []string{"dense", "rnn", "lstm", "cnn", "batch_norm", "dropout"}

// Possible activation functions
var possibleActivations = []string{"relu", "sigmoid", "tanh", "leaky_relu", "elu", "linear", "smooth_relu", "wavelet_act", "cauchy_act", "asym_act"}

// AddRandomNeuron adds a new neuron of the given type (or random type if empty) to the Phase.
// It creates random connections from existing neurons, sets a random bias, and chooses an activation if needed.
func (bp *Phase) AddRandomNeuron(neuronType string, activation string, minConnections, maxConnections int) *Neuron {
	// If neuronType is not provided, pick a random type
	if neuronType == "" {
		neuronType = neuronTypes[rand.Intn(len(neuronTypes))]
	}

	// If activation not provided, pick a random one
	if activation == "" {
		activation = possibleActivations[rand.Intn(len(possibleActivations))]
	}

	// Determine the new neuron's ID
	newID := bp.GetNextNeuronID()

	// Create a new neuron
	newNeuron := &Neuron{
		ID:         newID,
		Type:       neuronType,
		Bias:       rand.NormFloat64() * 0.1, // Small random bias
		Activation: activation,
	}

	// Determine how many connections to make
	numExisting := len(bp.Neurons)
	if numExisting == 0 {
		// If no existing neurons, just return the newly created neuron as is
		bp.Neurons[newID] = newNeuron
		return newNeuron
	}
	if minConnections < 1 {
		minConnections = 1
	}
	if maxConnections < minConnections {
		maxConnections = minConnections
	}
	numConns := rand.Intn(maxConnections-minConnections+1) + minConnections
	if numConns > numExisting {
		numConns = numExisting
	}

	// Get list of existing neuron IDs and shuffle
	existingIDs := bp.getAllNeuronIDs()
	rand.Shuffle(len(existingIDs), func(i, j int) { existingIDs[i], existingIDs[j] = existingIDs[j], existingIDs[i] })

	// Pick a subset of existing neurons to connect from
	selectedIDs := existingIDs[:numConns]

	// Create connections from selected neurons to the new neuron
	for _, sourceID := range selectedIDs {
		weight := rand.NormFloat64() * 0.1
		newNeuron.Connections = append(newNeuron.Connections, []float64{float64(sourceID), weight})
	}

	// Special handling for certain neuron types
	if neuronType == "lstm" {
		// Initialize gate weights
		conCount := len(newNeuron.Connections)
		newNeuron.GateWeights = map[string][]float64{
			"input":  bp.RandomWeights(conCount),
			"forget": bp.RandomWeights(conCount),
			"output": bp.RandomWeights(conCount),
			"cell":   bp.RandomWeights(conCount),
		}
	} else if neuronType == "cnn" && len(newNeuron.Kernels) == 0 {
		// Initialize default kernels if none provided
		newNeuron.Kernels = [][]float64{
			{rand.Float64(), rand.Float64()},
			{rand.Float64(), rand.Float64()},
		}
	} else if neuronType == "batch_norm" && newNeuron.BatchNormParams == nil {
		newNeuron.BatchNormParams = &BatchNormParams{
			Gamma: 1.0,
			Beta:  0.0,
			Mean:  0.0,
			Var:   1.0,
		}
	}

	// Add neuron to Phase
	bp.Neurons[newID] = newNeuron

	return newNeuron
}

// RewireOutputsThroughNewNeuron ensures that the newly added neuron influences the output neurons.
// One approach: connect the new neuron to all existing output neurons,
// so that their value now depends also on the new neuron.
// Optionally, you can remove some existing direct connections to outputs to force dependency.
// RewireOutputsThroughNewNeuron ensures the newly added neuron is
// the *only* path from the old pre-output neurons to the outputs.
func (bp *Phase) RewireOutputsThroughNewNeuron(newNeuronID int) {
	for _, outID := range bp.OutputNodes {
		outNeuron := bp.Neurons[outID]
		var newConns [][]float64
		// Keep only connections coming from neurons already marked as new
		for _, conn := range outNeuron.Connections {
			srcID := int(conn[0])
			// If the source neuron is marked as new, keep its connection
			if bp.Neurons[srcID].IsNew {
				newConns = append(newConns, conn)
			}
		}
		// Add a connection from the new neuron if it doesn't already exist.
		if !bp.connectionExists(newNeuronID, outID) {
			weight := rand.NormFloat64() * 0.1 // small random weight
			newConns = append(newConns, []float64{float64(newNeuronID), weight})
			if bp.Debug {
				fmt.Printf("Added connection from new neuron %d to output neuron %d with weight %f\n", newNeuronID, outID, weight)
			}
		}
		outNeuron.Connections = newConns
	}
}

// GetNextNeuronID finds the highest ID and increments it
// Rename to start with a capital letter to export it
func (bp *Phase) GetNextNeuronID() int {
	maxID := 0
	for id := range bp.Neurons {
		if id > maxID {
			maxID = id
		}
	}
	for id := range bp.QuantumNeurons {
		if id > maxID {
			maxID = id
		}
	}
	return maxID + 1
}

// AddConnection adds a new connection between two random neurons.
func (bp *Phase) AddConnection() {
	sourceID, targetID := bp.getRandomConnectionPair()
	if sourceID == -1 || targetID == -1 {
		return
	}
	weight := rand.NormFloat64() * 0.1
	bp.Neurons[targetID].Connections = append(bp.Neurons[targetID].Connections, []float64{float64(sourceID), weight})
	if bp.Debug {
		fmt.Printf("Added connection from Neuron %d to Neuron %d (weight=%f)\n", sourceID, targetID, weight)
	}
}

// RemoveConnection removes a random connection from a random neuron.
func (bp *Phase) RemoveConnection() {
	neuronIDs := bp.getAllNeuronIDs()
	if len(neuronIDs) == 0 {
		return
	}
	neuronID := neuronIDs[rand.Intn(len(neuronIDs))]
	neuron := bp.Neurons[neuronID]
	if len(neuron.Connections) == 0 {
		return
	}
	connIndex := rand.Intn(len(neuron.Connections))
	removedConn := neuron.Connections[connIndex]
	neuron.Connections = append(neuron.Connections[:connIndex], neuron.Connections[connIndex+1:]...)
	if bp.Debug {
		fmt.Printf("Removed connection from Neuron %d to Neuron %d\n", int(removedConn[0]), neuronID)
	}
}

// AdjustWeights modifies the weights of a random neuron's connections.
func (bp *Phase) AdjustWeights() {
	neuronIDs := bp.getAllNeuronIDs()
	if len(neuronIDs) == 0 {
		return
	}
	neuronID := neuronIDs[rand.Intn(len(neuronIDs))]
	neuron := bp.Neurons[neuronID]
	if len(neuron.Connections) == 0 {
		return
	}
	for i := range neuron.Connections {
		adjustment := rand.NormFloat64() * 0.05
		neuron.Connections[i][1] += adjustment
	}
	if bp.Debug {
		fmt.Printf("Adjusted weights for Neuron %d\n", neuronID)
	}
}

// AdjustBiases modifies the bias of a random neuron.
func (bp *Phase) AdjustBiases() {
	neuronIDs := bp.getAllNeuronIDs()
	if len(neuronIDs) == 0 {
		return
	}
	neuronID := neuronIDs[rand.Intn(len(neuronIDs))]
	neuron := bp.Neurons[neuronID]
	adjustment := rand.NormFloat64() * 0.05
	neuron.Bias += adjustment
	if bp.Debug {
		fmt.Printf("Adjusted bias for Neuron %d by %f\n", neuronID, adjustment)
	}
}

// ChangeActivationFunction changes the activation function of a random non-output neuron.
func (bp *Phase) ChangeActivationFunction() {
	nonOutputNeurons := []int{}
	for id := range bp.Neurons {
		if !contains(bp.OutputNodes, id) {
			nonOutputNeurons = append(nonOutputNeurons, id)
		}
	}
	if len(nonOutputNeurons) == 0 {
		return
	}
	neuronID := nonOutputNeurons[rand.Intn(len(nonOutputNeurons))]
	neuron := bp.Neurons[neuronID]
	possibleActivations := []string{"relu", "sigmoid", "tanh", "leaky_relu", "elu", "linear"}
	newAct := possibleActivations[rand.Intn(len(possibleActivations))]
	neuron.Activation = newAct
	if bp.Debug {
		fmt.Printf("Changed activation function of Neuron %d to %s\n", neuronID, newAct)
	}
}

// AdjustAllWeights adjusts all connection weights by the specified amount.
func (bp *Phase) AdjustAllWeights(adjustment float64) {
	for _, neuron := range bp.Neurons {
		for i := range neuron.Connections {
			neuron.Connections[i][1] += adjustment
		}
	}
	if bp.Debug {
		fmt.Printf("Adjusted all weights by %f\n", adjustment)
	}
}

// AdjustAllBiases adjusts all biases by the specified amount.
func (bp *Phase) AdjustAllBiases(adjustment float64) {
	for _, neuron := range bp.Neurons {
		neuron.Bias += adjustment
	}
	if bp.Debug {
		fmt.Printf("Adjusted all biases by %f\n", adjustment)
	}
}

// changeNeuronTypeTo changes the type of the neuron with the given ID to newType.
func (bp *Phase) changeNeuronTypeTo(neuronID int, newType string) {
	neuron, exists := bp.Neurons[neuronID]
	if !exists || neuron.Type == "input" {
		return
	}
	oldType := neuron.Type
	neuron.Type = newType

	// Reset type-specific fields
	neuron.GateWeights = nil
	neuron.Kernels = nil
	neuron.BatchNormParams = nil
	neuron.DropoutRate = 0
	neuron.CellState = 0

	// Initialize new type-specific fields
	switch newType {
	case "lstm":
		conCount := len(neuron.Connections)
		neuron.GateWeights = map[string][]float64{
			"input":  bp.RandomWeights(conCount),
			"forget": bp.RandomWeights(conCount),
			"output": bp.RandomWeights(conCount),
			"cell":   bp.RandomWeights(conCount),
		}
		neuron.CellState = 0
	case "cnn":
		neuron.Kernels = [][]float64{
			{rand.Float64(), rand.Float64()},
			{rand.Float64(), rand.Float64()},
		}
	case "batch_norm":
		neuron.BatchNormParams = &BatchNormParams{
			Gamma: 1.0,
			Beta:  0.0,
			Mean:  0.0,
			Var:   1.0,
		}
	case "dropout":
		neuron.DropoutRate = 0.5
	}

	if bp.Debug {
		fmt.Printf("Changed Neuron %d from %s to %s\n", neuronID, oldType, newType)
	}
}

// changeNeuronType changes the type of the neuron with the given ID to a random type different from its current type.
func (bp *Phase) changeNeuronType(neuronID int) {
	neuron, exists := bp.Neurons[neuronID]
	if !exists || neuron.Type == "input" {
		return
	}
	currentType := neuron.Type
	var possibleTypes []string
	for _, t := range neuronTypes {
		if t != currentType {
			possibleTypes = append(possibleTypes, t)
		}
	}
	if len(possibleTypes) == 0 {
		return
	}
	newType := possibleTypes[rand.Intn(len(possibleTypes))]
	bp.changeNeuronTypeTo(neuronID, newType)
}

// ChangeSingleNeuronType randomly selects one non-input neuron and changes its type to a different random type.
func (bp *Phase) ChangeSingleNeuronType() {
	nonInputNeurons := bp.getNonInputNeuronIDs()
	if len(nonInputNeurons) == 0 {
		if bp.Debug {
			fmt.Println("No non-input neurons available to change.")
		}
		return
	}
	neuronID := nonInputNeurons[rand.Intn(len(nonInputNeurons))]
	bp.changeNeuronType(neuronID)
}

// ChangePercentageOfNeuronsTypes changes the types of a specified percentage of non-input neurons to random types.
func (bp *Phase) ChangePercentageOfNeuronsTypes(percentage float64) {
	nonInputNeurons := bp.getNonInputNeuronIDs()
	total := len(nonInputNeurons)
	if total == 0 {
		if bp.Debug {
			fmt.Println("No non-input neurons available to change.")
		}
		return
	}
	if percentage < 0 {
		percentage = 0
	} else if percentage > 100 {
		percentage = 100
	}
	numToChange := int(float64(total) * (percentage / 100.0))
	if numToChange < 1 && percentage > 0 {
		numToChange = 1
	}
	rand.Shuffle(len(nonInputNeurons), func(i, j int) {
		nonInputNeurons[i], nonInputNeurons[j] = nonInputNeurons[j], nonInputNeurons[i]
	})
	for i := 0; i < numToChange && i < total; i++ {
		bp.changeNeuronType(nonInputNeurons[i])
	}
}

// RandomizeAllNeuronsTypes changes all non-input neurons to random types different from their current types.
func (bp *Phase) RandomizeAllNeuronsTypes() {
	nonInputNeurons := bp.getNonInputNeuronIDs()
	if len(nonInputNeurons) == 0 {
		if bp.Debug {
			fmt.Println("No non-input neurons available to randomize.")
		}
		return
	}
	for _, id := range nonInputNeurons {
		bp.changeNeuronType(id)
	}
}

func (bp *Phase) SetAllNeuronsToSameRandomType() {
	nonInputNeurons := bp.getNonInputNeuronIDs()
	if len(nonInputNeurons) == 0 {
		if bp.Debug {
			fmt.Println("No non-input neurons available to change.")
		}
		return
	}
	newType := neuronTypes[rand.Intn(len(neuronTypes))]
	for _, id := range nonInputNeurons {
		bp.changeNeuronTypeTo(id, newType)
	}
	if bp.Debug {
		fmt.Printf("Set all non-input neurons to type %s\n", newType)
	}
}
