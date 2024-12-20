package blueprint

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// MethodInfo represents metadata about a method, including its name, parameters, and parameter types.
type MethodInfo struct {
	MethodName string          `json:"method_name"`
	Parameters []ParameterInfo `json:"parameters"`
}

// ParameterInfo represents metadata about a parameter, including its name and type.
type ParameterInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// GetBlueprintMethodsJSON returns a JSON string containing all methods attached to the Blueprint struct,
// including each method's parameters and their types.
func (bp *Blueprint) GetBlueprintMethodsJSON() (string, error) {
	// Retrieve all methods and their metadata
	methods, err := bp.GetBlueprintMethods()
	if err != nil {
		return "", fmt.Errorf("failed to retrieve methods: %w", err)
	}

	// Convert methods metadata to JSON
	data, err := json.MarshalIndent(methods, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to serialize methods to JSON: %w", err)
	}

	return string(data), nil
}

// GetBlueprintMethods retrieves all methods of the Blueprint struct, including their names, parameters, and types.
func (bp *Blueprint) GetBlueprintMethods() ([]MethodInfo, error) {
	var methods []MethodInfo

	// Use reflection to inspect the Blueprint's methods
	bpType := reflect.TypeOf(bp)
	for i := 0; i < bpType.NumMethod(); i++ {
		method := bpType.Method(i)

		// Collect parameter information for each method
		var params []ParameterInfo
		methodType := method.Type
		for j := 1; j < methodType.NumIn(); j++ { // Start from 1 to skip the receiver
			paramType := methodType.In(j)
			param := ParameterInfo{
				Name: fmt.Sprintf("param%d", j),
				Type: paramType.String(),
			}
			params = append(params, param)
		}

		// Append method information
		methods = append(methods, MethodInfo{
			MethodName: method.Name,
			Parameters: params,
		})
	}

	return methods, nil
}
