package phase

import (
	"fmt"
	"math/rand"
)

// NewPhaseWithLayers creates a fully connected feedforward network with specified layer sizes.
func NewPhaseWithLayers(layers []int, hiddenAct, outputAct string) *Phase {
	bp := NewPhase()
	neuronID := 0

	// Add input neurons
	for i := 0; i < layers[0]; i++ {
		bp.Neurons[neuronID] = &Neuron{ID: neuronID, Type: "input"}
		bp.InputNodes = append(bp.InputNodes, neuronID)
		neuronID++
	}

	// Add hidden and output layers
	prevLayerStart := 0
	for l := 1; l < len(layers); l++ {
		currentLayerStart := neuronID
		for i := 0; i < layers[l]; i++ {
			act := hiddenAct
			if l == len(layers)-1 {
				act = outputAct
			}
			bp.Neurons[neuronID] = &Neuron{
				ID:         neuronID,
				Type:       "dense",
				Activation: act,
				Bias:       rand.Float64()*0.1 - 0.05,
			}
			// Connect to all neurons in the previous layer
			for j := prevLayerStart; j < currentLayerStart; j++ {
				weight := rand.Float64()*2 - 1
				bp.Neurons[j].Connections = append(bp.Neurons[j].Connections, []float64{float64(neuronID), weight})
				bp.Neurons[neuronID].Connections = append(bp.Neurons[neuronID].Connections, []float64{float64(j), weight})
			}
			if l == len(layers)-1 {
				bp.OutputNodes = append(bp.OutputNodes, neuronID)
			}
			neuronID++
		}
		prevLayerStart = currentLayerStart
	}

	return bp
}

// InitializeWithLayers resets the Phase and builds a fully connected network
// with the specified layers, hidden activation, and output activation.
func (bp *Phase) InitializeWithLayers(layers []int, hiddenAct, outputAct string) {
	bp.Neurons = make(map[int]*Neuron)
	bp.InputNodes = []int{}
	bp.OutputNodes = []int{}
	bp.QuantumNeurons = make(map[int]*QuantumNeuron)
	bp.ScalarActivationMap = scalarActivationFunctions
	bp.InitializeActivationFunctions()

	if bp.Debug {
		fmt.Printf("Initializing network with layers: %v, hiddenAct: %s, outputAct: %s\n", layers, hiddenAct, outputAct)
	}

	neuronID := 0
	for i := 0; i < layers[0]; i++ {
		bp.Neurons[neuronID] = &Neuron{ID: neuronID, Type: "input"}
		bp.InputNodes = append(bp.InputNodes, neuronID)
		if bp.Debug {
			fmt.Printf("Added input neuron %d\n", neuronID)
		}
		neuronID++
	}
	prevLayerStart := 0
	for l := 1; l < len(layers); l++ {
		currentLayerStart := neuronID
		for i := 0; i < layers[l]; i++ {
			act := hiddenAct
			if l == len(layers)-1 {
				act = outputAct
			}
			bp.Neurons[neuronID] = &Neuron{
				ID:         neuronID,
				Type:       "dense",
				Activation: act,
				Bias:       rand.Float64()*0.1 - 0.05,
			}
			for j := prevLayerStart; j < currentLayerStart; j++ {
				weight := rand.Float64()*2 - 1
				bp.Neurons[j].Connections = append(bp.Neurons[j].Connections, []float64{float64(neuronID), weight})
				bp.Neurons[neuronID].Connections = append(bp.Neurons[neuronID].Connections, []float64{float64(j), weight})
			}
			if l == len(layers)-1 {
				bp.OutputNodes = append(bp.OutputNodes, neuronID)
			}
			if bp.Debug {
				fmt.Printf("Added neuron %d (Type: %s, Activation: %s)\n", neuronID, bp.Neurons[neuronID].Type, act)
			}
			neuronID++
		}
		prevLayerStart = currentLayerStart
	}
	if bp.Debug {
		fmt.Printf("Initialization complete. Neurons: %d, InputNodes: %v, OutputNodes: %v\n", len(bp.Neurons), bp.InputNodes, bp.OutputNodes)
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
	return newBP
}
