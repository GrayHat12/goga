package brain

const EXPORT_VERSION uint = 1

type ConnectionExport struct {
	From     string  `json:"from"`
	To       string  `json:"to"`
	Strength float64 `json:"strength"`
}

type NodeExport struct {
	Id          string             `json:"id"`
	Weight      float64            `json:"weight"`
	Bias        float64            `json:"bias"`
	Connections []ConnectionExport `json:"connections"`
}

type BrainExport struct {
	Version     uint         `json:"version"`
	Id          string       `json:"id"`
	InputNodes  []NodeExport `json:"inputNodes"`
	HiddenNodes []NodeExport `json:"hiddenNodes"`
	OutputNodes []NodeExport `json:"outputNodes"`
}
