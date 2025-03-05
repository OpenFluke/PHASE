package phase

import (
	"fmt"
	"math/rand"
)

// NewPhaseWithLayers creates a strictly feed-forward network
// with the given layer sizes. For example, []int{784, 64, 10} means
// 784 input neurons, one hidden layer of 64 neurons, and 10 output neurons.
func NewPhaseWithLayers(layers []int, hiddenAct, outputAct string) *Phase {
	bp := NewPhase()
	neuronID := 0

	// 1) Create input neurons (layer 0)
	for i := 0; i < layers[0]; i++ {
		bp.Neurons[neuronID] = &Neuron{
			ID:   neuronID,
			Type: "input",
		}
		bp.InputNodes = append(bp.InputNodes, neuronID)
		neuronID++
	}

	// 2) Create all subsequent layers (hidden + final output layer)
	prevLayerStart := 0
	prevLayerEnd := layers[0] // i.e. next ID after the last input neuron

	for layerIndex := 1; layerIndex < len(layers); layerIndex++ {
		layerSize := layers[layerIndex]
		currentLayerStart := neuronID
		for i := 0; i < layerSize; i++ {
			// Decide activation: hidden vs. output
			act := hiddenAct
			if layerIndex == len(layers)-1 {
				act = outputAct
			}
			// Create new neuron
			bp.Neurons[neuronID] = &Neuron{
				ID:         neuronID,
				Type:       "dense",
				Activation: act,
				Bias:       rand.Float64()*0.1 - 0.05, // small random bias
			}
			// Only forward connections from the previous layer *to* this neuron
			for srcID := prevLayerStart; srcID < prevLayerEnd; srcID++ {
				w := rand.Float64()*2 - 1
				bp.Neurons[neuronID].Connections = append(
					bp.Neurons[neuronID].Connections,
					[]float64{float64(srcID), w},
				)
			}

			// If it’s the final layer, mark as output
			if layerIndex == len(layers)-1 {
				bp.OutputNodes = append(bp.OutputNodes, neuronID)
			}

			neuronID++
		}
		// Advance “previous layer” window
		prevLayerStart = currentLayerStart
		prevLayerEnd = currentLayerStart + layerSize
	}

	return bp
}

// InitializeWithLayers resets this Phase and builds a strictly feed-forward network
// with the specified layers, hidden activation, and output activation.
func (bp *Phase) InitializeWithLayers(layers []int, hiddenAct, outputAct string) {
	// Wipe the existing Phase maps/slices
	bp.Neurons = make(map[int]*Neuron)
	bp.InputNodes = []int{}
	bp.OutputNodes = []int{}
	bp.QuantumNeurons = make(map[int]*QuantumNeuron)
	bp.ScalarActivationMap = scalarActivationFunctions
	bp.InitializeActivationFunctions()

	if bp.Debug {
		fmt.Printf("Initializing feed-forward network with layers: %v, hiddenAct: %s, outputAct: %s\n",
			layers, hiddenAct, outputAct)
	}

	neuronID := 0

	// 1) Create input neurons (layer 0)
	for i := 0; i < layers[0]; i++ {
		bp.Neurons[neuronID] = &Neuron{
			ID:   neuronID,
			Type: "input",
		}
		bp.InputNodes = append(bp.InputNodes, neuronID)
		if bp.Debug {
			fmt.Printf("Added input neuron %d\n", neuronID)
		}
		neuronID++
	}

	// 2) Create hidden + output layers
	prevLayerStart := 0
	prevLayerEnd := layers[0]
	for layerIndex := 1; layerIndex < len(layers); layerIndex++ {
		layerSize := layers[layerIndex]
		currentLayerStart := neuronID
		for i := 0; i < layerSize; i++ {
			// Activation depends on whether it's the final layer
			act := hiddenAct
			if layerIndex == len(layers)-1 {
				act = outputAct
			}
			// Create the neuron
			bp.Neurons[neuronID] = &Neuron{
				ID:         neuronID,
				Type:       "dense",
				Activation: act,
				Bias:       rand.Float64()*0.1 - 0.05,
			}
			// Add forward connections from previous layer
			for srcID := prevLayerStart; srcID < prevLayerEnd; srcID++ {
				w := rand.Float64()*2 - 1
				bp.Neurons[neuronID].Connections = append(
					bp.Neurons[neuronID].Connections,
					[]float64{float64(srcID), w},
				)
			}

			// If final layer, add to outputs
			if layerIndex == len(layers)-1 {
				bp.OutputNodes = append(bp.OutputNodes, neuronID)
			}
			if bp.Debug {
				fmt.Printf("Added neuron %d (Type: %s, Activation: %s)\n",
					neuronID, bp.Neurons[neuronID].Type, act)
			}
			neuronID++
		}
		prevLayerStart = currentLayerStart
		prevLayerEnd = currentLayerStart + layerSize
	}

	if bp.Debug {
		fmt.Printf("Initialization complete.\n"+
			"Total neurons: %d\nInputNodes: %v\nOutputNodes: %v\n",
			len(bp.Neurons), bp.InputNodes, bp.OutputNodes)
	}
}

// Copy creates a deep copy of the Phase instance.
func (bp *Phase) Copy() *Phase {
	data, err := bp.SerializeToJSON()
	if err != nil {
		panic(fmt.Sprintf("failed to serialize Phase for copying: %v", err))
	}
	newBP := NewPhase()
	err = newBP.DeserializesFromJSON(data)
	if err != nil {
		panic(fmt.Sprintf("failed to deserialize Phase for copying: %v", err))
	}
	newBP.ID = bp.GetNextPhaseID() // Assign a new unique ID
	return newBP
}

// GetNextPhaseID generates a unique ID for new Phase instances (simple increment for this example)
func (bp *Phase) GetNextPhaseID() int {
	// This is a simple implementation; in a real scenario, track globally or use a counter
	return rand.Intn(10000) + 1 // Random ID for demo; replace with proper counter if needed
}
