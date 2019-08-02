package fonet

import (
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
	tests := []struct {
		name    string
		samples [][][]float64
		results [][][]float64
	}{
		{
			name:    "Output range 0-1",
			samples: samples,
			results: [][][]float64{
				{{0, 0}, {0}},
				{{0, 1}, {1}},
				{{1, 0}, {1}},
				{{1, 1}, {0}},
			},
		},
		{
			name:    "Output range 10-20",
			samples: samples,
			results: [][][]float64{
				{{0, 0}, {10}},
				{{0, 1}, {20}},
				{{1, 0}, {20}},
				{{1, 1}, {10}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t1 *testing.T) {
			n, err := NewNetwork([]int{2, 3, 1})
			if err != nil {
				t1.Fatalf("Could not create network: %+v", err)
			}

			n.Train(tt.samples, 10000, 1.01, false)

			for _, exp := range tt.results {
				if res := n.Predict(exp[0])[0]; !percDiffLessThan(res, exp[1][0], 2) {
					t1.Errorf("Result is too different to be accurate; Got: %.2f, expected: %.2f", res, exp[1][0])
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
