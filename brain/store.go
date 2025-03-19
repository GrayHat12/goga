package brain

type NodeType int

const (
	INPUT_NODE NodeType = iota
	HIDDEN_NODE
	OUTPUT_NODE
)

type Session struct {
	Nodes  []GNode
	Brains []Brain
	Config *Config
}

func NewSession(config *Config) *Session {
	if config == nil {
		config = LoadDefaultConfig()
	}
	return &Session{
		Nodes:  []GNode{},
		Brains: []Brain{},
		Config: config,
	}
}

func (session *Session) NewNodeId(node GNode) int {
	session.Nodes = append(session.Nodes, node)
	return len(session.Nodes)
}

func (session *Session) NewBrainId(brain Brain) int {
	session.Brains = append(session.Brains, brain)
	return len(session.Brains)
}
