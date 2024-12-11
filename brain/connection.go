package brain

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/GrayHat12/goga/utils"
)

type Connection struct {
	threadLock *sync.Mutex
	strength   float64
	from       *Node
	to         *Node
}

func NewConnection(from, to *Node) *Connection {
	connection := &Connection{
		threadLock: &sync.Mutex{},
		strength:   utils.GaussianRandom(nil),
		from:       from,
		to:         to,
	}
	connection.from.AddOutgoingConnection(*connection)
	connection.to.AddOutgoingConnection(*connection)
	return connection
}

func (connection *Connection) GetId() string {
	return fmt.Sprintf("connection-%s:%s", connection.from.id, connection.to.id)
}

func (connection *Connection) SetStrength(strength float64) {
	connection.threadLock.Lock()
	defer connection.threadLock.Unlock()
	connection.strength = strength
}

func (connection *Connection) GetOutput() float64 {
	val := connection.from.GetOutput() * connection.strength
	if connection.from.nodeType == INPUT {
		return val
	} else {
		return utils.Tanh(val)
	}
}

func (connection *Connection) Update(from, to *Node) {
	connection.threadLock.Lock()
	defer connection.threadLock.Unlock()
	connection.from = from
	connection.to.RemoveIncomingConnection(connection)
	connection.to = to
}

func (connection *Connection) Mutate() {
	connection.threadLock.Lock()
	defer connection.threadLock.Unlock()
	if rand.Float64() < CONNECTION_STRENGTH_MUTATE_PROBABILITY {
		connection.strength += utils.GaussianRandom(nil) * CONNECTION_STRENGTH_MUTATION_SCOPE
	}
}
