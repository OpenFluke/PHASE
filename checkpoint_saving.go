package phase

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

func (bp *Phase) EnsureCheckpointFolder(CheckpointFolder string) error {
	if _, err := os.Stat(CheckpointFolder); os.IsNotExist(err) {
		return os.MkdirAll(CheckpointFolder, 0755)
	}
	return nil
}

// SaveCheckpoint saves a single checkpoint for a specific sample index
func (bp *Phase) SaveCheckpoint(CheckpointFolder string, sampleIndex int, checkpoint map[int]map[string]interface{}) error {
	// Ensure the checkpoint folder exists
	if err := bp.EnsureCheckpointFolder(CheckpointFolder); err != nil {
		return fmt.Errorf("failed to ensure checkpoint folder: %w", err)
	}

	// Construct the file path (e.g., "checkpoints/sample_0.json")
	fileName := filepath.Join(CheckpointFolder, fmt.Sprintf("sample_%d.json", sampleIndex))

	// Marshal the checkpoint data to JSON with indentation for readability
	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint for sample %d: %w", sampleIndex, err)
	}

	// Write the JSON data to the file
	if err := ioutil.WriteFile(fileName, data, 0644); err != nil {
		return fmt.Errorf("failed to write checkpoint file %s: %w", fileName, err)
	}

	// Optional debug logging (assumes a Debug flag exists in the package)
	if bp.Debug {
		fmt.Printf("Saved checkpoint for sample %d to %s\n", sampleIndex, fileName)
	}

	return nil
}

// LoadCheckpoint loads a single checkpoint for a specific sample index
func (bp *Phase) LoadCheckpoint(CheckpointFolder string, sampleIndex int) (map[int]map[string]interface{}, error) {
	// Construct the file path
	fileName := filepath.Join(CheckpointFolder, fmt.Sprintf("sample_%d.json", sampleIndex))

	// Read the checkpoint file
	data, err := ioutil.ReadFile(fileName)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint file %s: %w", fileName, err)
	}

	// Unmarshal the JSON data into a checkpoint structure
	var checkpoint map[int]map[string]interface{}
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint for sample %d: %w", sampleIndex, err)
	}

	return checkpoint, nil
}
