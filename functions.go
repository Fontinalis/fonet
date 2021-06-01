package fonet

import "math"

// Activation functions available to this package.  These are taken from
// https://en.wikipedia.org/wiki/Activation_function
const (
	Sigmond ActivationFunction = iota
	BentIdentity
	ReLU
	LeakyReLU
	ArSinH
)

var (
	functionPairs = map[ActivationFunction]funcPair{
		Sigmond:      {sigmoid, sigmoidD},
		BentIdentity: {bentIdent, bentIdentD},
		ReLU:         {reLU, reLUD},
		LeakyReLU:    {leakyReLU, leakyReLUD},
		ArSinH:       {arSinH, arSinHD},
	}
)

// ActivationFunction is the type of function to use for the neural network.
type ActivationFunction int

func (a ActivationFunction) String() string {
	switch a {
	case Sigmond:
		return "Sigmond"
	case BentIdentity:
		return "Bent Identity"
	case ReLU:
		return "Rectified linear unit"
	case LeakyReLU:
		return "Leaky rectified linear unit"
	case ArSinH:
		return "ArSinH"
	default:
		return "Unknown"
	}
}

type funcPair [2]func(float64) float64

func sigmoid(z float64) float64 {
	return 1. / (1. + math.Exp(-z))
}

func sigmoidD(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}

func bentIdent(z float64) float64 {
	numerator := (math.Sqrt((z * z) + 1)) - 1
	denominator := 2.0

	return (numerator / denominator) + z
}

func bentIdentD(z float64) float64 {
	numerator := z
	denominator := 2 * math.Sqrt((z*z)+1)

	return (numerator / denominator) + 1
}

func reLU(z float64) float64 {
	if z > 0 {
		return z
	}

	return 0
}

func reLUD(z float64) float64 {
	if z > 0 {
		return 1
	}
	return 0
}

func leakyReLU(z float64) float64 {
	if z < 0 {
		return 0.01 * z
	}
	return z
}

func leakyReLUD(z float64) float64 {
	if z < 0 {
		return 0.01
	}
	return 1
}

func arSinH(z float64) float64 {
	return math.Log(z + math.Sqrt((z*z)+1))
}

func arSinHD(z float64) float64 {
	return 1 / math.Sqrt((z*z)+1)
}
