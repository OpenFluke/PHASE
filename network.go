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
