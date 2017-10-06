package fonet

import "testing"

var samples [][][]float64 = [][][]float64{
	[][]float64{
		[]float64{
			0,
			0,
		},
		[]float64{
			0,
		},
	},
	[][]float64{
		[]float64{
			0,
			1,
		},
		[]float64{
			1,
		},
	},
	[][]float64{
		[]float64{
			1,
			0,
		},
		[]float64{
			1,
		},
	},
	[][]float64{
		[]float64{
			1,
			1,
		},
		[]float64{
			0,
		},
	},
}

func TestNetwork(t *testing.T) {
	n, err := NewNetwork([]int{2, 3, 1})
	if err != nil {
		t.FailNow()
	}

	n.Train(samples, 10000, 1.01, false)

	a := n.Predict([]float64{0, 0})[0]
	b := n.Predict([]float64{0, 1})[0]
	c := n.Predict([]float64{1, 0})[0]
	d := n.Predict([]float64{1, 1})[0]
	if int(a+0.5) != 0 || int(b+0.5) != 1 || int(c+0.5) != 1 || int(d+0.5) != 0 {
		t.Fail()
	}
}
