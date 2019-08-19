package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	poyntingg  = NewVectorField("poynting", "", "Poynting vector", setpoynting)
)

//###################
//Poynting vector
func setpoynting(dst *data.Slice) {
	poynting(dst, DU, norm_stress.Quantity, shear_stress.Quantity)
}

func poynting(dst *data.Slice, du firstDerivative, NormS, ShearS Quantity) {
	normS := ValueOf(NormS)
	defer cuda.Recycle(normS)

	shearS := ValueOf(ShearS)
	defer cuda.Recycle(shearS)
	cuda.Poynting(dst, du.Buffer(), normS, shearS, U.Mesh())
}