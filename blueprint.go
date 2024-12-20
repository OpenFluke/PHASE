package blueprint

import (
	"fmt"
	"math/rand"
)

// Blueprint encapsulates the entire neural network
type Blueprint struct {
	Neurons             map[int]*Neuron           `json:"neurons"`
	QuantumNeurons      map[int]*QuantumNeuron    `json:"quant"`
	InputNodes          []int                     `json:"input_nodes"`
	OutputNodes         []int                     `json:"output_nodes"`
	ScalarActivationMap map[string]ActivationFunc `json:"-"`
	Debug               bool                      `json:"-"`
}

// ModelMetadata holds metadata, evaluation benchmarks, and additional information for models in the AI framework.
type ModelMetadata struct {
	// Basic model information
	ModelID           string   `json:"modelID"`
	ProjectName       string   `json:"projectName"`
	Description       string   `json:"description,omitempty"` // Optional description for the model
	ParentModelIDs    []string `json:"parentModelIDs"`
	ChildModelIDs     []string `json:"childModelIDs"`
	CreationTimestamp string   `json:"creationTimestamp"`
	LastModified      string   `json:"lastModified"`

	// Neuron and layer information
	TotalNeurons int64  `json:"totalNeurons"`
	TotalLayers  int64  `json:"totalLayers"`
	LayerRange   [2]int `json:"layerRange"`  // Min and max layers
	NeuronRange  [2]int `json:"neuronRange"` // Min and max neurons per layer

	// Accuracy and error metrics
	LastTrainingAccuracy              float64 `json:"lastTrainingAccuracy"`
	LastTestAccuracy                  float64 `json:"lastTestAccuracy"`
	LastTestAccuracyGenerous          float64 `json:"lastTestAccuracyGenerous"`
	LastTestAccuracyForgiveness       float64 `json:"lastTestAccuracyForgiveness"`
	ForgivenessThreshold              float64 `json:"forgivenessThreshold"`
	LastTrainingExactErrorCount       int64   `json:"lastTrainingExactErrorCount"`
	LastTestExactErrorCount           int64   `json:"lastTestExactErrorCount"`
	LastTrainingAverageGenerousError  float64 `json:"lastTrainingAverageGenerousError"`
	LastTestAverageGenerousError      float64 `json:"lastTestAverageGenerousError"`
	LastTrainingForgivenessErrorCount int64   `json:"lastTrainingForgivenessErrorCount"`
	LastTestForgivenessErrorCount     int64   `json:"lastTestForgivenessErrorCount"`

	// Training and testing session information
	//TrainingSessions []TrainingSession `json:"trainingSessions"`
	//TestingSessions  []TestingSession  `json:"testingSessions"`

	// Evaluation and performance benchmarks
	//BenchmarkResults BenchmarkResults `json:"benchmarkResults"`
	Evaluated bool   `json:"evaluated"`
	Path      string `json:"path"`

	// Model mutation and adjustment settings
	PossibleMutations         []string `json:"possibleMutations"`
	BiasAdjustmentIncrement   float64  `json:"biasAdjustmentIncrement"`
	WeightAdjustmentIncrement float64  `json:"weightAdjustmentIncrement"`

	// Advanced metadata
	OptimizedFor           string   `json:"optimizedFor,omitempty"` // E.g., "speed", "accuracy", "efficiency"
	CompatibleEnvironments []string `json:"compatibleEnvironments"` // Supported deployment environments (e.g., "desktop", "web", "cloud")
	Tags                   []string `json:"tags,omitempty"`         // Tags for categorizing models

	// Extended neuron and processing information
	NeuronTypes         []string `json:"neuronTypes"`         // List of neuron types used (e.g., Dense, CNN, RNN)
	AttentionMechanisms bool     `json:"attentionMechanisms"` // Whether attention mechanisms are included
	DropoutUsed         bool     `json:"dropoutUsed"`         // Whether dropout layers are used

	// Resource requirements
	EstimatedMemoryUsage string `json:"estimatedMemoryUsage,omitempty"` // Approximate memory usage
	EstimatedComputeTime string `json:"estimatedComputeTime,omitempty"` // Estimated compute time for typical runs
}

// NewBlueprint creates and initializes a new Blueprint
func NewBlueprint() *Blueprint {
	bp := &Blueprint{
		Neurons:             make(map[int]*Neuron),
		InputNodes:          []int{},
		QuantumNeurons:      make(map[int]*QuantumNeuron),
		OutputNodes:         []int{},
		ScalarActivationMap: scalarActivationFunctions,
	}
	bp.InitializeActivationFunctions()
	return bp
}

// blueprint.go

// RandomWeights generates random weights for connections
func (bp *Blueprint) RandomWeights(size int) []float64 {
	weights := make([]float64, size)
	for i := range weights {
		weights[i] = rand.NormFloat64() * 0.5 // Increase scale
	}
	return weights
}

// AddInputNodes adds multiple input nodes to the network
func (bp *Blueprint) AddInputNodes(ids []int) {
	bp.InputNodes = append(bp.InputNodes, ids...)
}

// AddOutputNodes adds multiple output nodes to the network
func (bp *Blueprint) AddOutputNodes(ids []int) {
	bp.OutputNodes = append(bp.OutputNodes, ids...)
}

// ApplyScalarActivation applies the specified scalar activation function
func (bp *Blueprint) ApplyScalarActivation(value float64, activation string) float64 {
	if actFunc, exists := bp.ScalarActivationMap[activation]; exists {
		return actFunc(value)
	}
	// Log a warning and use linear activation
	if bp.Debug {
		fmt.Printf("Warning: Undefined activation '%s'. Using linear activation.\n", activation)
	}
	return Linear(value)
}

// Forward propagates inputs through the network
// Forward propagates inputs through the network
func (bp *Blueprint) Forward(inputs map[int]float64, timesteps int) {
	// Set input neurons
	for id, value := range inputs {
		if neuron, exists := bp.Neurons[id]; exists {
			neuron.Value = value
			if bp.Debug {
				fmt.Printf("Input Neuron %d set to %f\n", id, value)
			}
		}
	}

	// Process neurons over timesteps
	for t := 0; t < timesteps; t++ {
		if bp.Debug {
			fmt.Printf("=== Timestep %d ===\n", t)
		}

		// Process all neurons, including hidden neurons
		for id := 1; id <= len(bp.Neurons); id++ {
			neuron, exists := bp.Neurons[id]
			if !exists || neuron.Type == "input" { // Skip input neurons
				continue
			}

			// Gather inputs from connected neurons
			inputValues := []float64{}
			for _, conn := range neuron.Connections {
				sourceID := int(conn[0])
				weight := conn[1]
				if sourceNeuron, exists := bp.Neurons[sourceID]; exists {
					inputValues = append(inputValues, sourceNeuron.Value*weight)
				}
			}

			// Process the neuron
			bp.ProcessNeuron(neuron, inputValues, t)
		}
	}

	// Apply softmax to output neurons
	bp.ApplySoftmax()
}

// RunNetwork runs the neural network with given inputs and timesteps
func (bp *Blueprint) RunNetwork(inputs map[int]float64, timesteps int) {
	bp.Forward(inputs, timesteps)
	if bp.Debug {
		outputs := bp.GetOutputs()
		fmt.Println("Final Outputs:")
		for id, value := range outputs {
			fmt.Printf("Neuron %d: %f\n", id, value)
		}
	}
}

// GetOutputs retrieves the output values from the network
func (bp *Blueprint) GetOutputs() map[int]float64 {
	outputs := make(map[int]float64)
	for _, id := range bp.OutputNodes {
		if neuron, exists := bp.Neurons[id]; exists {
			outputs[id] = neuron.Value
		}
	}
	return outputs
}
