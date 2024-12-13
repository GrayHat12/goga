package brain

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/GrayHat12/goslice/commons"
)

var BRAIN_INDEX = 0

var brainIndexLock = sync.Mutex{}

func GetNewBrainId() int {
	brainIndexLock.Lock()
	defer brainIndexLock.Unlock()
	BRAIN_INDEX += 1
	return BRAIN_INDEX
}

type Brain struct {
	threadLock  *sync.Mutex
	inputNodes  []Node
	hiddenNodes []Node
	outputNode  Node
	id          string
}

func NewBrain(inputCount int) *Brain {
	if inputCount <= 1 {
		panic("invalid input count")
	}
	brain := Brain{
		threadLock: &sync.Mutex{},
		id:         fmt.Sprintf("Network-%d", GetNewBrainId()),
		inputNodes: []Node{},
		hiddenNodes: []Node{
			*NewNode(HIDDEN), *NewNode(HIDDEN),
		},
		outputNode: *NewNode(OUTPUT),
	}
	for range inputCount {
		node := NewNode(INPUT)

		NewConnection(node, &brain.hiddenNodes[0])
		NewConnection(node, &brain.hiddenNodes[1])
		brain.inputNodes = append(brain.inputNodes, *node)
	}
	NewConnection(&brain.hiddenNodes[0], &brain.outputNode)
	NewConnection(&brain.hiddenNodes[1], &brain.outputNode)
	return &brain
}

func (brain *Brain) FeedForward(inputs []float64) float64 {
	brain.threadLock.Lock()
	defer brain.threadLock.Unlock()
	for index, item := range brain.inputNodes {
		if index < len(inputs) {
			item.UpdateInput(inputs[index])
		} else {
			item.UpdateInput(0)
		}
	}
	return brain.outputNode.GetOutput()
}

func (brain *Brain) Mutate() {
	brain.threadLock.Lock()
	defer brain.threadLock.Unlock()
	mutateNode := func(ctx context.Context, node *Node, _ int, _ *[]Node) {
		select {
		case <-ctx.Done():
			return
		default:
			if rand.Float64() < MUTATION_PROBABILITY {
				node.Mutate()
				commons.ForEach(context.Background(), &node.outgoingConnections, func(ctx context.Context, x *Connection, _ int, _ *[]Connection) {
					select {
					case <-ctx.Done():
						return
					default:
						x.Mutate()
					}
				})
				if rand.Float64() < CONNECTION_SPLIT_PROBABILITY*(5.0-float64(len(brain.hiddenNodes))/3.0) {
					newNode := NewNode(HIDDEN)
					newNode.SetWeight(1)
					newNode.SetBias(0)
					connectionToSplit := node.outgoingConnections[int(math.Floor(rand.Float64()*float64(len(node.outgoingConnections))))]
					connection := NewConnection(newNode, connectionToSplit.to)
					connection.SetStrength(1)
					connectionToSplit.Update(node, newNode)
					brain.hiddenNodes = append(brain.hiddenNodes, *newNode)
				}
			}
		}
	}
	commons.ForEach(context.Background(), &brain.inputNodes, mutateNode)
	commons.ForEach(context.Background(), &brain.hiddenNodes, mutateNode)

	if rand.Float64() < NEW_CONNECTION_PROBABILITY {
		possibleNodeRange1 := len(brain.inputNodes) + len(brain.hiddenNodes)
		randomPick1Index := int(math.Floor(rand.Float64() * float64(possibleNodeRange1)))
		var randomPick1 Node
		if randomPick1Index < len(brain.inputNodes) {
			randomPick1 = brain.inputNodes[randomPick1Index]
		} else {
			randomPick1 = brain.hiddenNodes[randomPick1Index-len(brain.inputNodes)]
		}

		// possibleNodes2 := utils.Filter(brain.hiddenNodes, func(x Node, _ int, _ []Node) bool {
		// 	return x.id != randomPick1.id && !randomPick1.IsInvalidChildNode(&x)
		// })
		possibleNodes2 := []*Node{}
		validNodeChannel := make(chan *Node, len(brain.hiddenNodes))
		commons.ForEach(context.Background(), &brain.hiddenNodes, func(ctx context.Context, x *Node, _ int, _ *[]Node) {
			select {
			case <-ctx.Done():
				return
			default:
				if x.id != randomPick1.id && !randomPick1.IsInvalidChildNode(x) {
					validNodeChannel <- x
				}
			}
		})
		for range len(brain.hiddenNodes) {
			possibleNodes2 = append(possibleNodes2, <-validNodeChannel)
		}
		close(validNodeChannel)
		if len(possibleNodes2) > 0 {
			randomPick2 := possibleNodes2[int(math.Floor(rand.Float64()*float64(len(possibleNodes2))))]
			NewConnection(&randomPick1, randomPick2)
		}
	}
}
