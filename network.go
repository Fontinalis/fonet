package fonet

import (
	"errors"
	"log"
	"math"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Network is containing all the needed settings/variables.
type Network struct {
	w      [][][]float64           // weights
	b      [][]float64             // biases
	d      [][]float64             // delta values for
	z      [][]float64             // z values in each layer
	l      int                     // Number of the layers
	ls     []int                   // number of the neurons in each layer
	aFunc  func(z float64) float64 // activation function
	daFunc func(z float64) float64 // derivative of the aFunc
}

func sigmoid(z float64) float64 {
	return 1. / (1. + math.Exp(-z))
}

func sigmoidD(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

// NewNetwork is for creating a new network
// with the defined layers.
func NewNetwork(ls []int) (*Network, error) {
	if len(ls) < 3 {
		return nil, errors.New("Not enough layer in the layers description")
	}
	n := Network{
		l:      len(ls) - 1,
		ls:     ls[1:],
		aFunc:  sigmoid,
		daFunc: sigmoidD,
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

// Train is for training the network with the specified dataset,
// epoch and learning rate
// The last bool parameter is for tracking where the training is. It'll log each epoch.
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
	_ = n.feedforward(x) // define z values

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

// use only in the backpropagation! othervise it can return wrong value
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

// Predict calculates the output for the given input
func (n *Network) Predict(input []float64) []float64 {
	return n.feedforward(input)
}
