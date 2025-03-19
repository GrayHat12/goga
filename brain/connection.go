package brain

import (
	"fmt"
	"math/rand"

	"github.com/GrayHat12/goga/maths"
)

type GConnection struct {
	session  *Session
	strength float64
	from     *GNode
	to       *GNode
}

func NewConnection(session *Session, from *GNode, to *GNode) GConnection {
	connection := GConnection{
		session:  session,
		strength: maths.GaussianRandom(0, 1),
		from:     from,
		to:       to,
	}
	connection.from.AddOutgoingConnection(connection)
	connection.to.AddIncomingConnection(connection)
	return connection
}

func (connection GConnection) GetStrength() float64 {
	return connection.strength
}

func (connection GConnection) GetFrom() *GNode {
	return connection.from
}

func (connection GConnection) GetTo() *GNode {
	return connection.to
}

func (connection GConnection) GetId() string {
	return fmt.Sprintf("connection-%s:%s", connection.from.GetId(), connection.to.GetId())
}

func (connection *GConnection) SetStrength(strength float64) {
	connection.strength = strength
}

func (connection GConnection) GetOutput() float64 {
	val := connection.from.GetOutput() * connection.strength
	if connection.from.GetNodeType() == INPUT_NODE {
		return val
	} else {
		return maths.Tanh(val)
	}
}

func (connection *GConnection) UpdateConnection(from *GNode, to *GNode) {
	connection.from = from
	connection.to.RemoveIncomingConnection(connection)
	connection.to = to
}

func (connection *GConnection) Mutate() {
	if rand.Float64() < connection.session.Config.CONNECTION_STRENGTH_MUTATE_PROBABILITY {
		connection.strength += maths.GaussianRandom(0, 1) * connection.session.Config.CONNECTION_STRENGTH_MUTATION_SCOPE
	}
}
