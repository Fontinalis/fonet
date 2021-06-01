package fonet

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

var (
	// ErrNotEnoughLayers is returned when trying to create a new network with too few layers.
	ErrNotEnoughLayers = errors.New("too few layers, minimum of 3 required")
)

// Network is containing all the needed settings/variables.
type Network struct {
	// w is the weights of the network.
	w [][][]float64
	// b is the biases of the network.
	b [][]float64
	// d is the delta values for each layer.
	d [][]float64
	// z is the z values for each layer.
	z [][]float64
	// l holds the number of layers in the network.
	l int
	// ls is the number of neurons in each layer.
	ls []int
	// activationID is the ID of the activation function used.  This is stored for serialization purposes.
	activationID ActivationFunction
	// aFunc is the activation function
	aFunc func(z float64) float64
	// daFunc is the derivative of the activation function.
	daFunc func(z float64) float64
}

type jsonNetwork struct {
	W            [][][]float64 `json:"W"`
	B            [][]float64   `json:"B"`
	D            [][]float64   `json:"D"`
	Z            [][]float64   `json:"Z"`
	L            int           `json:"L"`
	LS           []int         `json:"LS"`
	ActivationID int           `json:"ActivationID"`
}

// MarshalJSON implements the Marshaler interface for JSON encoding.
func (n *Network) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	err := json.NewEncoder(&buf).Encode(jsonNetwork{
		W:            n.w,
		B:            n.b,
		D:            n.d,
		Z:            n.z,
		L:            n.l,
		LS:           n.ls,
		ActivationID: int(n.activationID),
	})
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// UnmarshalJSON implements the Unmarshaler interface for JSON encoding.
func (n *Network) UnmarshalJSON(data []byte) error {
	var en jsonNetwork
	if err := json.Unmarshal(data, &en); err != nil {
		return err
	}

	n.w = en.W
	n.b = en.B
	n.d = en.D
	n.z = en.Z
	n.l = en.L
	n.ls = en.LS
	n.activationID = ActivationFunction(en.ActivationID)
	n.aFunc = functionPairs[n.activationID][0]
	n.daFunc = functionPairs[n.activationID][1]

	return nil
}

// NewNetwork is for creating a new network with the defined layers.
func NewNetwork(ls []int, activationFunc ActivationFunction) (*Network, error) {
	if len(ls) < 3 {
		return nil, ErrNotEnoughLayers
	}

	n := Network{
		l:            len(ls) - 1,
		ls:           ls[1:],
		activationID: activationFunc,
		aFunc:        functionPairs[activationFunc][0],
		daFunc:       functionPairs[activationFunc][1],
	}

	// init weights
	n.w = make([][][]float64, n.l)
	n.w[0] = make([][]float64, ls[0])
	for i := 0; i < ls[0]; i++ {
		n.w[0][i] = make([]float64, n.ls[0])
		for j := 0; j < n.ls[0]; j++ {
			n.w[0][i][j] = rand.Float64()
		}
	}
	for l := 1; l < n.l; l++ {
		n.w[l] = make([][]float64, n.ls[l-1])
		for i := 0; i < n.ls[l-1]; i++ {
			n.w[l][i] = make([]float64, n.ls[l])
			for j := 0; j < n.ls[l]; j++ {
				n.w[l][i][j] = rand.Float64()
			}
		}
	}

	// init biases, deltas, z(s)
	n.b = make([][]float64, n.l)
	n.d = make([][]float64, n.l)
	n.z = make([][]float64, n.l)
	for l := 0; l < n.l; l++ {
		n.b[l] = make([]float64, n.ls[l])
		for i := 0; i < n.ls[l]; i++ {
			n.b[l][i] = rand.Float64()
		}
		n.d[l] = make([]float64, n.ls[l])
		n.z[l] = make([]float64, n.ls[l])
	}

	return &n, nil
}

func (n *Network) dw(l, i, j int, eta float64) float64 {
	return -eta * n.d[l][j] * n.a(l-1, i)
}

func (n *Network) a(l, j int) float64 {
	return n.aFunc(n.z[l][j])
}

// Train is for training the network with the specified dataset, epoch and learning rate. The last bool parameter is for
// tracking where the training is. It'll log each epoch.
func (n *Network) Train(trainingData [][][]float64, epochs int, lrate float64, debug bool) {
	for e := 0; e < epochs; e++ {
		for _, xy := range trainingData {
			n.backpropagate(xy, lrate)
		}
		if debug {
			log.Println("Epoch:", e+1, "/", epochs)
		}
	}
}

func (n *Network) backpropagate(xy [][]float64, eta float64) {
	x := xy[0]
	y := xy[1]
	// define z values
	_ = n.feedforward(x)

	// define the output deltas
	for j := 0; j < len(n.d[len(n.d)-1]); j++ {
		n.d[len(n.d)-1][j] = (n.a(len(n.d)-1, j) - y[j]) * n.daFunc(n.z[len(n.d)-1][j])
	}

	// define the inner deltas
	for l := len(n.d) - 2; l >= 0; l-- {
		for j := 0; j < len(n.d[l]); j++ {
			n.d[l][j] = n.delta(l, j)
		}
	}

	// update weights
	for i := 0; i < len(n.w[0]); i++ {
		for j := 0; j < len(n.w[0][i]); j++ {
			n.w[0][i][j] += -eta * n.d[0][j] * x[i]
		}
	}
	for l := 1; l < len(n.w); l++ {
		for i := 0; i < len(n.w[l]); i++ {
			for j := 0; j < len(n.w[l][i]); j++ {
				n.w[l][i][j] += n.dw(l, i, j, eta)
			}
		}
	}

	// update biases
	for l := 0; l < len(n.b); l++ {
		for j := 0; j < len(n.b[l]); j++ {
			n.b[l][j] += -eta * n.d[l][j]
		}
	}
}

// delta should only use in the back-propagation, otherwise it can return wrong value.
func (n *Network) delta(l, j int) float64 {
	var d float64
	for k := 0; k < n.ls[l+1]; k++ {
		d += n.d[l+1][k] * n.w[l+1][j][k] * n.daFunc(n.z[l][j])
	}
	return d
}

func (n *Network) feedforward(a []float64) []float64 {
	for l := 0; l < n.l; l++ {
		atemp := make([]float64, n.ls[l])
		for j := 0; j < n.ls[l]; j++ {
			n.z[l][j] = 0
			for i := 0; i < len(a); i++ {
				n.z[l][j] += n.w[l][i][j] * a[i]
			}
			n.z[l][j] += n.b[l][j]
			atemp[j] = n.aFunc(n.z[l][j])
		}
		a = atemp
	}
	return a
}

// Predict calculates the output for the given input.
func (n *Network) Predict(input []float64) []float64 {
	return n.feedforward(input)
}

// Export will serialize the network, and write it to the provided writer.
func (n *Network) Export(w io.Writer) error {
	return json.NewEncoder(w).Encode(n)
}

// Load will load a network from the provided reader.
func Load(r io.Reader) (*Network, error) {
	var n Network
	if err := json.NewDecoder(r).Decode(&n); err != nil {
		return nil, err
	}

	return &n, nil
}

// LoadFrom will load a network from the provided byte slice.
func LoadFrom(bs []byte) (*Network, error) {
	buf := bytes.NewBuffer(bs)
	return Load(buf)
}
