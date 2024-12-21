package blueprint

import (
	"fmt"
	"math/rand"
)

// Possible neuron types for mutation
var neuronTypes = []string{"dense", "rnn", "lstm", "cnn", "batch_norm", "dropout", "nca"}

// Possible activation functions
var possibleActivations = []string{"relu", "sigmoid", "tanh", "leaky_relu", "elu", "linear"}

// AddRandomNeuron adds a new neuron of the given type (or random type if empty) to the blueprint.
// It creates random connections from existing neurons, sets a random bias, and chooses an activation if needed.
func (bp *Blueprint) AddRandomNeuron(neuronType string, activation string, minConnections, maxConnections int) *Neuron {
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

	// Add neuron to blueprint
	bp.Neurons[newID] = newNeuron

	return newNeuron
}

// RewireOutputsThroughNewNeuron ensures that the newly added neuron influences the output neurons.
// One approach: connect the new neuron to all existing output neurons,
// so that their value now depends also on the new neuron.
// Optionally, you can remove some existing direct connections to outputs to force dependency.
func (bp *Blueprint) RewireOutputsThroughNewNeuron(newNeuronID int) {
	for _, outID := range bp.OutputNodes {
		// Before adding a connection, ensure it doesn't already exist
		if !bp.connectionExists(newNeuronID, outID) {
			// Add a new connection from the new neuron to the output neuron
			weight := rand.NormFloat64() * 0.1
			outNeuron := bp.Neurons[outID]
			outNeuron.Connections = append(outNeuron.Connections, []float64{float64(newNeuronID), weight})
			if bp.Debug {
				fmt.Printf("Added connection from Neuron %d to Output Neuron %d (weight=%f)\n", newNeuronID, outID, weight)
			}
		}
	}
}

// GetNextNeuronID finds the highest ID and increments it
// Rename to start with a capital letter to export it
func (bp *Blueprint) GetNextNeuronID() int {
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

