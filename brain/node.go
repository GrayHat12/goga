package brain

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/GrayHat12/goga/utils"
	"github.com/GrayHat12/goslice/commons"
	"github.com/GrayHat12/goslice/inplace"
)

var NODE_INDEX = 0

var nodeIndexLock = sync.Mutex{}

func GetNewNodeId() int {
	nodeIndexLock.Lock()
	defer nodeIndexLock.Unlock()
	NODE_INDEX += 1
	return NODE_INDEX
}

type NodeType int

const (
	INPUT NodeType = iota
	HIDDEN
	OUTPUT
)

type Node struct {
	threadLock          *sync.Mutex
	weight              float64
	bias                float64
	outgoingConnections []Connection
	id                  string
	incomingConnections []Connection
	input               float64
	nodeType            NodeType
}

func NewNode(nodeType NodeType) *Node {
	if nodeType == OUTPUT {
		return &Node{
			threadLock:          &sync.Mutex{},
			weight:              1,
			bias:                0,
			outgoingConnections: []Connection{},
			id:                  fmt.Sprintf("node-%d", GetNewNodeId()),
			incomingConnections: []Connection{},
			input:               0,
			nodeType:            nodeType,
		}
	} else {
		return &Node{
			threadLock:          &sync.Mutex{},
			weight:              utils.GaussianRandom(nil),
			bias:                utils.GaussianRandom(nil) / 4,
			outgoingConnections: []Connection{},
			id:                  fmt.Sprintf("node-%d", GetNewNodeId()),
			incomingConnections: []Connection{},
			input:               0,
			nodeType:            nodeType,
		}
	}
}

func (node *Node) AddOutgoingConnection(connection Connection) {
	node.threadLock.Lock()
	defer node.threadLock.Unlock()
	node.outgoingConnections = append(node.outgoingConnections, connection)
}

// func (node *Node) GetWeight() float64 {
// 	return node.weight
// }

// func (node *Node) GetBias() float64 {
// 	return node.bias
// }

// func (node *Node) GetOutgoingConnections() *[]Connection {
// 	return &node.outgoingConnections
// }

// func (node *Node) GetId() string {
// 	return node.id
// }

// func (node *Node) GetNodeType() NodeType {
// 	return node.nodeType
// }

func (node *Node) SetWeight(weight float64) {
	node.threadLock.Lock()
	defer node.threadLock.Unlock()
	node.weight = weight
}

func (node *Node) SetBias(bias float64) {
	node.threadLock.Lock()
	defer node.threadLock.Unlock()
	node.bias = bias
}

func (node *Node) Mutate() {
	node.threadLock.Lock()
	defer node.threadLock.Unlock()
	if rand.Float64() < NODE_WEIGHT_MUTATE_PROBABILITY {
		node.weight += utils.GaussianRandom(nil) * NODE_WEIGHT_MUTATION_SCOPE
	}
	if rand.Float64() < NODE_BIAS_MUTATE_PROBABILITY {
		node.bias += utils.GaussianRandom(nil) * NODE_BIAS_MUTATION_SCOPE
	}
}

func (node *Node) UpdateInput(value float64) {
	node.threadLock.Lock()
	defer node.threadLock.Unlock()
	node.input = value
}

func (node *Node) GetOutput() float64 {
	val := 0.0
	if len(node.incomingConnections) > 0 {
		sumOfInputs := 0.0
		for _, item := range node.incomingConnections {
			sumOfInputs += item.GetOutput()
		}
		val = (sumOfInputs * node.weight) + node.bias
	} else {
		val = (node.input * node.weight) + node.bias
	}
	if node.nodeType == OUTPUT {
		return utils.Sigmoid(val)
	} else {
		return val
	}
}

func (node *Node) AddIncomingConnection(connection Connection) {
	node.threadLock.Lock()
	defer node.threadLock.Unlock()
	node.incomingConnections = append(node.incomingConnections, connection)
}

func (node *Node) RemoveIncomingConnection(connection *Connection) {
	node.threadLock.Lock()
	defer node.threadLock.Unlock()
	node.incomingConnections = *inplace.Filter(&node.incomingConnections, func(item *Connection, _ int, _ *[]Connection) bool {
		return item.GetId() != connection.GetId()
	})
}

func (node *Node) IsInvalidChildNode(child *Node) bool {
	if node.id == child.id {
		return true
	}
	if commons.Find(&node.incomingConnections, func(item *Connection, _ int, _ *[]Connection) bool {
		return item.from.id == node.id || item.from.IsInvalidChildNode(node)
	}) != nil {
		return true
	}
	return false
}
