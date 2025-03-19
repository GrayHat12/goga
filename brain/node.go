package brain

import (
	"fmt"
	"math/rand"

	"github.com/GrayHat12/goga/maths"
)

type GNode struct {
	session             *Session
	weight              float64
	bias                float64
	outgoingConnections []GConnection
	id                  string
	incomingConnections []GConnection
	input               float64
	nodeType            NodeType
}

func NewNode(session *Session, nodeType NodeType) GNode {
	bias := maths.GaussianRandom(0, 1) / 4
	weight := maths.GaussianRandom(0, 1)
	if nodeType == OUTPUT_NODE {
		bias = 0
		weight = 1
	}
	node := GNode{
		session:  session,
		bias:     bias,
		weight:   weight,
		nodeType: nodeType,
	}
	node.id = fmt.Sprintf("node-%d", session.NewNodeId(node))
	return node
}

func (node *GNode) AddOutgoingConnection(connection GConnection) {
	node.outgoingConnections = append(node.outgoingConnections, connection)
}

func (node *GNode) AddIncomingConnection(connection GConnection) {
	node.incomingConnections = append(node.incomingConnections, connection)
}

func (node *GNode) RemoveIncomingConnection(connection *GConnection) {
	newConnectionList := []GConnection{}
	for _, element := range node.incomingConnections {
		if connection == &element {
			continue
		} else {
			newConnectionList = append(newConnectionList, element)
		}
	}
	node.incomingConnections = newConnectionList
}

func (node GNode) GetWeight() float64 {
	return node.weight
}

func (node GNode) GetBias() float64 {
	return node.bias
}

func (node GNode) GetOutgoingConnections() []GConnection {
	return node.outgoingConnections
}

func (node GNode) GetId() string {
	return node.id
}

func (node GNode) GetNodeType() NodeType {
	return node.nodeType
}

func (node *GNode) SetWeight(weight float64) {
	node.weight = weight
}

func (node *GNode) SetBias(bias float64) {
	node.bias = bias
}

func (node *GNode) Mutate() {
	if rand.Float64() < node.session.Config.NODE_WEIGHT_MUTATE_PROBABILITY {
		node.weight += maths.GaussianRandom(0, 1) * node.session.Config.NODE_WEIGHT_MUTATION_SCOPE
	}
	if rand.Float64() < node.session.Config.NODE_BIAS_MUTATE_PROBABILITY {
		node.bias += maths.GaussianRandom(0, 1) * node.session.Config.NODE_BIAS_MUTATION_SCOPE
	}
}

func (node *GNode) UpdateInput(value float64) {
	node.input = value
}

func (node *GNode) GetOutput() float64 {
	val := 0.0
	if len(node.incomingConnections) > 0 {
		sumOfInputs := 0.0
		for _, connection := range node.incomingConnections {
			sumOfInputs += connection.GetOutput()
		}
		val = (sumOfInputs * node.weight) + node.bias
	} else {
		val = (node.input * node.weight) + node.bias
	}
	if node.nodeType == OUTPUT_NODE {
		return maths.Sigmoid(val)
	} else {
		return val
	}
}

func (node GNode) IsInvalidChildNode(other *GNode) bool {
	if &node == other {
		return true
	}
	if node.id == other.GetId() {
		return true
	}
	for _, connection := range node.incomingConnections {
		if connection.GetFrom().GetId() == node.id || connection.GetFrom().IsInvalidChildNode(&node) {
			return true
		}
	}
	return false
}
