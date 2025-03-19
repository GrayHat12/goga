package brain

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/GrayHat12/goga/maths"
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
