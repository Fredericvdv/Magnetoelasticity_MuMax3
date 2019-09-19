package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	SecondDerivDisp = NewVectorField("force", "", "Force/volume", calcSecondDerivDisp)
	C11             = NewScalarParam("C11", "N/m2", "Stiffness constant C11")
	C12             = NewScalarParam("C12", "N/m2", "Stiffness constant C12")
	C44             = NewScalarParam("C44", "N/m2", "Stiffness constant C44")
)

func calcSecondDerivDisp(dst *data.Slice) {
	SecondDerivative(dst, U, C11, C12, C44)
}

func SecondDerivative(dst *data.Slice, u displacement, C1, C2, C3 *RegionwiseScalar) {
	c1 := C1.MSlice()
	defer c1.Recycle()

	c2 := C2.MSlice()
	defer c2.Recycle()

	c3 := C3.MSlice()
	defer c3.Recycle()
	cuda.SecondDerivative(dst, u.Buffer(), U.Mesh(), c1, c2, c3)
}
