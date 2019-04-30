package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

//Als deze variabele opgeropen wordt in het Mumax script, dan wordt de functie calcSecondDerivMag uitgevoerd
var (
	SecondDerivDisp = NewVectorField("force", "", "Force/volume", calcSecondDerivDisp)
	C1              = NewScalarParam("C1", "N/m2", "Stiffness constant 1")
	C2              = NewScalarParam("C2", "N/m2", "Stiffness constant 2")
	C3              = NewScalarParam("C3", "N/m2", "Stiffness constant 3")
	rho             = NewScalarParam("rho", "kg/m3", "Density")
)

func calcSecondDerivDisp(dst *data.Slice) {
	SecondDerivative(dst, U, C1, C2, C3)
}

//regelmatig make runnen = compilen van wat je nu hebt op je branch
// voor je begint altijd eerst in je branch gaan: git checkout "branchname"
//Capital letter to call them from outsit the package engine
func SecondDerivative(dst *data.Slice, u displacement, C1, C2, C3 *RegionwiseScalar) {
	//cuda kernel aanroepen
	//m.Buffer() neemt de data.Slice uit de magnetization m
	//M=magnetizatie --> mesh wordt hiervan genomen
	if C1.nonZero() || C2.nonZero() || C3.nonZero() {
		c1 := C1.MSlice()
		defer c1.Recycle()

		c2 := C2.MSlice()
		defer c2.Recycle()

		c3 := C3.MSlice()
		defer c3.Recycle()
		cuda.SecondDerivative(dst, u.Buffer(), U.Mesh(), c1, c2, c3)
	}
}
