package brain

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/GrayHat12/goga/maths"
	"github.com/GrayHat12/goslice/outofplace"
)

type Brain struct {
	lock        *sync.Mutex
	session     *Session
	inputNodes  []GNode
	hiddenNodes []GNode
	outputNodes []GNode
	id          string
}

func NewBrain(session *Session, inputCount int, outputCount int) *Brain {
	brain := Brain{
		lock:        &sync.Mutex{},
		session:     session,
		inputNodes:  []GNode{},
		hiddenNodes: []GNode{},
		outputNodes: []GNode{},
	}
	brain.id = fmt.Sprintf("Brain-%d", session.NewBrainId(brain))
	brain.hiddenNodes = append(brain.hiddenNodes, NewNode(session, HIDDEN_NODE))
	for range inputCount {
		node := NewNode(session, INPUT_NODE)
		NewConnection(session, &node, &brain.hiddenNodes[0])
		brain.inputNodes = append(brain.inputNodes, node)
	}
	for range outputCount {
		node := NewNode(session, OUTPUT_NODE)
		NewConnection(session, &brain.hiddenNodes[0], &node)
		brain.outputNodes = append(brain.outputNodes, node)
	}
	return &brain
}

func (brain *Brain) FeedForward(inputs []float64) []float64 {
	brain.lock.Lock()
	defer brain.lock.Unlock()
	for index := range brain.inputNodes {
		if index < len(inputs) {
			brain.inputNodes[index].UpdateInput(inputs[index])
		} else {
			brain.inputNodes[index].UpdateInput(0.0)
		}
	}
	output := []float64{}
	for _, node := range brain.outputNodes {
		output = append(output, node.GetOutput())
	}
	return output
}

func (brain *Brain) Mutate() {
	brain.lock.Lock()
	defer brain.lock.Unlock()
	nodeMutate := func(node *GNode) {
		if rand.Float64() < MUTATION_PROBABILITY {
			node.Mutate()
			for _, connection := range node.outgoingConnections {
				connection.Mutate()
			}
			if rand.Float64() < CONNECTION_SPLIT_PROBABILITY*(5.0-float64(len(brain.hiddenNodes))/3.0) {
				newNode := NewNode(brain.session, HIDDEN_NODE)
				newNode.SetWeight(1)
				newNode.SetBias(0)
				connectionToSplit := node.outgoingConnections[maths.FloorInt(rand.Float64()*float64(len(node.outgoingConnections)))]
				connection := NewConnection(brain.session, &newNode, connectionToSplit.to)
				connection.SetStrength(1)
				connectionToSplit.UpdateConnection(node, &newNode)
				brain.hiddenNodes = append(brain.hiddenNodes, newNode)
			}
		}
	}
	for _, node := range brain.inputNodes {
		nodeMutate(&node)
	}
	for _, node := range brain.hiddenNodes {
		nodeMutate(&node)
	}
	if rand.Float64() < NEW_CONNECTION_PROBABILITY {
		possibleNodes1 := []*GNode{}
		for _, node := range brain.inputNodes {
			possibleNodes1 = append(possibleNodes1, &node)
		}
		for _, node := range brain.hiddenNodes {
			possibleNodes1 = append(possibleNodes1, &node)
		}

		randomPick1 := possibleNodes1[maths.FloorInt(rand.Float64()*float64(len(possibleNodes1)))]
		possibleNodes2 := []*GNode{}
		for _, node := range brain.hiddenNodes {
			if node.id != randomPick1.id && !randomPick1.IsInvalidChildNode(&node) {
				possibleNodes2 = append(possibleNodes2, &node)
			}
		}

		if len(possibleNodes2) > 0 {
			randomPick2 := possibleNodes2[maths.FloorInt(rand.Float64()*float64(len(possibleNodes2)))]
			NewConnection(brain.session, randomPick1, randomPick2)
		}
	}
}

func (brain Brain) GetId() string {
	brain.lock.Lock()
	defer brain.lock.Unlock()
	return brain.id
}

func (brain Brain) CountHiddenNodes() int {
	return len(brain.hiddenNodes)
}

func (brain Brain) CountInputNdes() int {
	return len(brain.inputNodes)
}

func (brain Brain) CountOutputNodes() int {
	return len(brain.outputNodes)
}

func (brain Brain) Export() *BrainExport {
	brain.lock.Lock()
	defer brain.lock.Unlock()
	nodeMap := func(node *GNode, _ int, _ *[]GNode) *NodeExport {
		return &NodeExport{
			Id:     node.id,
			Weight: node.weight,
			Bias:   node.bias,
			Connections: outofplace.Map(node.outgoingConnections, func(connection *GConnection, _ int, _ *[]GConnection) *ConnectionExport {
				return &ConnectionExport{
					From:     connection.from.id,
					To:       connection.to.id,
					Strength: connection.strength,
				}
			}),
		}
	}
	return &BrainExport{
		Version:     EXPORT_VERSION,
		Id:          brain.id,
		InputNodes:  outofplace.Map(brain.inputNodes, nodeMap),
		HiddenNodes: outofplace.Map(brain.hiddenNodes, nodeMap),
		OutputNodes: outofplace.Map(brain.outputNodes, nodeMap),
	}
}

func (brain *Brain) Import(exported *BrainExport) {
	brain.lock.Lock()
	defer brain.lock.Unlock()

	brain.inputNodes = []GNode{}
	brain.hiddenNodes = []GNode{}
	brain.outputNodes = []GNode{}

	nodeMapping := map[string]*GNode{}
	pendingConnections := map[string]*ConnectionExport{}

	// output node
	for _, outputNodeExport := range exported.OutputNodes {
		outputNode := NewNode(brain.session, OUTPUT_NODE)
		outputNode.SetWeight(outputNodeExport.Weight)
		outputNode.SetBias(outputNodeExport.Bias)

		nodeMapping[outputNodeExport.Id] = &outputNode

		brain.outputNodes = append(brain.outputNodes, outputNode)
	}

	// hidden node
	for _, hiddenNodeExport := range exported.HiddenNodes {
		hiddenNode := NewNode(brain.session, HIDDEN_NODE)
		hiddenNode.SetWeight(hiddenNodeExport.Weight)
		hiddenNode.SetBias(hiddenNodeExport.Bias)
		for _, connectionExport := range hiddenNodeExport.Connections {
			pendingConnections[fmt.Sprintf("%s-%s", connectionExport.From, connectionExport.To)] = &connectionExport
		}

		nodeMapping[hiddenNodeExport.Id] = &hiddenNode

		brain.hiddenNodes = append(brain.hiddenNodes, hiddenNode)
	}

	// input node
	for _, inputNodeExport := range exported.InputNodes {
		inputNode := NewNode(brain.session, INPUT_NODE)
		inputNode.SetWeight(inputNodeExport.Weight)
		inputNode.SetBias(inputNodeExport.Bias)
		for _, connectionExport := range inputNodeExport.Connections {
			pendingConnections[fmt.Sprintf("%s-%s", connectionExport.From, connectionExport.To)] = &connectionExport
		}

		nodeMapping[inputNodeExport.Id] = &inputNode

		brain.inputNodes = append(brain.inputNodes, inputNode)
	}

	// connections
	for _, pendingConnection := range pendingConnections {
		NewConnection(brain.session, nodeMapping[pendingConnection.From], nodeMapping[pendingConnection.To])
	}
}
