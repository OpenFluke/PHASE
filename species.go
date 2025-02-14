package blueprint

import "math"

// BlueprintDistance computes a simple distance metric between two blueprints.
// It calculates the average absolute difference in biases and connection weights
// for neurons with the same ID.
func BlueprintDistance(bp1, bp2 *Blueprint) float64 {
	totalDiff := 0.0
	count := 0.0

	for id, neuron1 := range bp1.Neurons {
		if neuron2, exists := bp2.Neurons[id]; exists {
			// Difference in bias
			totalDiff += math.Abs(neuron1.Bias - neuron2.Bias)
			count++

			// Compare connection weights for common connections
			minLen := len(neuron1.Connections)
			if len(neuron2.Connections) < minLen {
				minLen = len(neuron2.Connections)
			}
			for i := 0; i < minLen; i++ {
				totalDiff += math.Abs(neuron1.Connections[i][1] - neuron2.Connections[i][1])
				count++
			}
		}
	}
	if count == 0 {
		return math.Inf(1)
	}
	return totalDiff / count
}

// ClusterBlueprintsBySpecies groups blueprints into species based on a distance threshold.
// It returns a map where each key is a species ID and the value is a slice of blueprint IDs.
func ClusterBlueprintsBySpecies(blueprints map[int]*Blueprint, threshold float64) map[int][]int {
	speciesMapping := make(map[int][]int)
	// For each species, keep a representative blueprint (the first one encountered)
	speciesRepresentative := make(map[int]*Blueprint)
	nextSpeciesID := 1

	// Iterate over each blueprint
	for id, bp := range blueprints {
		assigned := false
		for speciesID, rep := range speciesRepresentative {
			if BlueprintDistance(bp, rep) < threshold {
				// Blueprint is similar to the representative of this species
				speciesMapping[speciesID] = append(speciesMapping[speciesID], id)
				assigned = true
				break
			}
		}
		if !assigned {
			// Create a new species if no existing one is similar enough
			speciesMapping[nextSpeciesID] = []int{id}
			speciesRepresentative[nextSpeciesID] = bp
			nextSpeciesID++
		}
	}
	return speciesMapping
}
