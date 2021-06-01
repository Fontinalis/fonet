# fonet

[![Go](https://github.com/Fontinalis/fonet/actions/workflows/go.yml/badge.svg?branch=master)](https://github.com/Fontinalis/fonet/actions/workflows/go.yml)
[![Coverage Status](https://coveralls.io/repos/github/Fontinalis/fonet/badge.svg?branch=master)](https://coveralls.io/github/Fontinalis/fonet?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/Fontinalis/fonet)](https://goreportcard.com/report/github.com/Fontinalis/fonet)
[![Go Reference](https://pkg.go.dev/badge/github.com/Fontinalis/fonet.svg)](https://pkg.go.dev/github.com/Fontinalis/fonet)

`fonet` is a deep neural network package for Go. It's mainly created because I wanted to learn about neural networks and create my own package. I'm planning to continue the development of the package and add more function to it, for example exporting/importing a model.

## Install

It's the same as everywhere, you just have to run the
```
go get github.com/Fontinalis/fonet
```

## Usage

I focused (and still focusing) on creating an easy to use package, but let me know if something is not clear.

### Creating a network
As in the `xor` example, it's not so complicated to create a network.
When you creating the network, you always have to define the layers.
```go
n := fonet.NewNetwork([]int{2, 3, 1}, fonet.Sigmond)
/*
2 nodes in the INPUT LAYER
3 nodes in the HIDDEN LAYER
1 node in the OUTPUT LAYER
*/
```
But my goal was also to create a package, which can create deep neural networks too, so here is another example for that.
```go
n := fonet.NewNetwork([]int{6, 12, 8, 4}, fonet.Sigmond)
/*
6 nodes in the INPUT LAYER
12 nodes in the HIDDEN LAYER (1)
8 nodes in the HIDDEN LAYER (2)
4 nodes in the OUTPUT LAYER
*/
```


### Train the network
After creating the network, you have to train your network. To do that, you have to specify your training set, which should be like the next
```go
var trainingData = [][][]float64{
    [][]float64{ // The actual training sample
        []float64{
            /*
            The INPUT data
            */
        },
        []float64{
            /*
            The OUTPUT data
            */
        },
    },
}
```
After giving the training data, you can set the epoch and the learning rate.
```go
n.Train(trainingData, epoch, lrate, true)
// Train(trainingData [][][]float64, epochs int, lrate float64, debug bool)
```
`
Note: When 'debug' is true, it'll show when and which epoch is finished
`
### Predict the output
After training your network, using the `Predict(..)` function you can calculate the output for the given input. 

In the case of XOR, it looks like the next
```go
input := []float64{
    1,
    1,
}
out := n.Predict(input)
```
