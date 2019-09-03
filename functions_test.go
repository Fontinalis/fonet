package fonet

import "testing"

func TestActivationFunction_String(t *testing.T) {
	tests := []struct {
		name string
		a    ActivationFunction
		want string
	}{
		{
			name: "Sigmond",
			a:    Sigmond,
			want: "Sigmond",
		},
		{
			name: "BentIdentity",
			a:    BentIdentity,
			want: "Bent Identity",
		},
		{
			name: "ReLU",
			a:    ReLU,
			want: "Rectified linear unit",
		},
		{
			name: "LeakyReLU",
			a:    LeakyReLU,
			want: "Leaky rectified linear unit",
		},
		{
			name: "ArSinH",
			a:    ArSinH,
			want: "ArSinH",
		},
		{
			name: "Invalid",
			a:    -1,
			want: "Unknown",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.a.String(); got != tt.want {
				t.Errorf("String() = %v, want %v", got, tt.want)
			}
		})
	}
}
