package fonet

import (
	"bytes"
	"io/ioutil"
	"log"
	"math"
	"testing"
)

var samples = [][][]float64{
	{
		{
			0,
			0,
		},
		{
			0,
		},
	},
	{
		{
			0,
			1,
		},
		{
			1,
		},
	},
	{
		{
			1,
			0,
		},
		{
			1,
		},
	},
	{
		{
			1,
			1,
		},
		{
			0,
		},
	},
}

var samples2 = [][][]float64{
	{
		{
			0,
			0,
		},
		{
			10,
		},
	},
	{
		{
			0,
			1,
		},
		{
			20,
		},
	},
	{
		{
			1,
			0,
		},
		{
			20,
		},
	},
	{
		{
			1,
			1,
		},
		{
			10,
		},
	},
}

func TestNetwork(t *testing.T) {
	log.SetOutput(ioutil.Discard)

	tests := []struct {
		name               string
		samples            [][][]float64
		results            [][][]float64
		activationFunction ActivationFunction
	}{
		{
			name:    "Output range 0-1 Sigmond",
			samples: samples,
			results: [][][]float64{
				{{0, 0}, {0}},
				{{0, 1}, {1}},
				{{1, 0}, {1}},
				{{1, 1}, {0}},
			},
			activationFunction: Sigmond,
		},
		// {
		// 	name:    "Output range 0-1 Bent Identity",
		// 	samples: samples,
		// 	results: [][][]float64{
		// 		{{0, 0}, {0}},
		// 		{{0, 1}, {1}},
		// 		{{1, 0}, {1}},
		// 		{{1, 1}, {0}},
		// 	},
		// 	activationFunction: BentIdentity,
		// },
		{
			name:    "Output range 0-1 Rectified linear unit",
			samples: samples,
			results: [][][]float64{
				{{0, 0}, {0}},
				{{0, 1}, {1}},
				{{1, 0}, {1}},
				{{1, 1}, {0}},
			},
			activationFunction: ReLU,
		},
		// {
		// 	name:    "Output range 0-1 Leaky rectified linear unit",
		// 	samples: samples,
		// 	results: [][][]float64{
		// 		{{0, 0}, {0}},
		// 		{{0, 1}, {1}},
		// 		{{1, 0}, {1}},
		// 		{{1, 1}, {0}},
		// 	},
		// 	activationFunction: LeakyReLU,
		// },
		// {
		// 	name:    "Output range 0-1 ArSinH",
		// 	samples: samples,
		// 	results: [][][]float64{
		// 		{{0, 0}, {0}},
		// 		{{0, 1}, {1}},
		// 		{{1, 0}, {1}},
		// 		{{1, 1}, {0}},
		// 	},
		// 	activationFunction: ArSinH,
		// },
		// {
		// 	name:    "Output range 10-20",
		// 	samples: samples2,
		// 	results: [][][]float64{
		// 		{{0, 0}, {10}},
		// 		{{0, 1}, {20}},
		// 		{{1, 0}, {20}},
		// 		{{1, 1}, {10}},
		// 	},
		// 	activationFunction: ???,
		// },
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t1 *testing.T) {
			// NewNetwork should error with too few layers
			n, err := NewNetwork([]int{1}, tt.activationFunction)
			if err != ErrNotEnoughLayers {
				t1.Fatal("Expected error, but received no error")
			}
			n, err = NewNetwork([]int{3, 1}, tt.activationFunction)
			if err != ErrNotEnoughLayers {
				t1.Fatal("Expected error, but received no error")
			}

			n, err = NewNetwork([]int{2, 3, 1}, tt.activationFunction)
			if err != nil {
				t1.Fatalf("Could not create network: %+v", err)
			}

			n.Train(tt.samples, 10000, 1.01, true)

			// Export and re-load network to ensure those functions are tested.
			var buf bytes.Buffer
			if err := n.Export(&buf); err != nil {
				t1.Fatalf("Unable to export network: %+v", err)
			}
			networkBytes := buf.Bytes()
			// Using LoadFrom tests both the Load and LoadFrom functions.
			nn, err := LoadFrom(networkBytes)
			if err != nil {
				t1.Fatalf("Could not load network: %+v", err)
			}

			for _, exp := range tt.results {
				if res := nn.Predict(exp[0])[0]; !percDiffLessThan(res, exp[1][0], 2) {
					t1.Errorf("Result is too different to be accurate; Using %s got: %.2f, expected: %.2f", tt.activationFunction, res, exp[1][0])
				}
			}
		})
	}
}

// percDiffLessThan returns whether v1 and v2 differ by the percentage.
func percDiffLessThan(v1, v2, perc float64) bool {
	absDiff := math.Abs(v1 - v2)
	// Prevent issues with divide by zero
	if absDiff == 0 || v1 == 0 || v2 == 0 {
		return true
	}

	decDiff := absDiff / math.Max(v1, v2)
	return decDiff*100.0 < perc
}
