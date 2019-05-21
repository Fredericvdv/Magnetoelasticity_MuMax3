package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Eta = NewScalarParam("eta", "kg/(sm3)", "Damping constant")
	Rho = NewScalarParam("rho", "kg/m3", "Density")
)

func calcRightPart(dst, f, g *data.Slice) {
	RightPart(dst, f, g, Eta, Rho)
}

func RightPart(dst, f, g *data.Slice, Eta, Rho *RegionwiseScalar) {
	//No elastodynamics is calculated if stiffness constants are zero
	if Rho.nonZero() {
		rho, _ := Rho.Slice()
		//defer rho.Free()
		defer cuda.Recycle(rho)

		eta, _ := Eta.Slice()
		//defer eta.Free()
		defer cuda.Recycle(eta)

		cuda.RightPart(dst, f, g, eta, rho)
	}
}
