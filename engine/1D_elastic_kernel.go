package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

//Als deze variabele opgeropen wordt in het Mumax script, dan wordt de functie calcSecondDerivMag uitgevoerd
var (
	SecondDerivDisp = NewVectorField("d2udx2", "", "SecondDeriv", calcSecondDerivDisp)
)

func calcSecondDerivDisp(dst *data.Slice) {
	SecondDerivative(dst, U)
}

//regelmatig make runnen = compilen van wat je nu hebt op je branch
// voor je begint altijd eerst in je branch gaan: git checkout "branchname"
//Capital letter to call them from outsit the package engine
func SecondDerivative(dst *data.Slice, u displacement) {
	//cuda kernel aanroepen
	//m.Buffer() neemt de data.Slice uit de magnetization m
	//M=magnetizatie --> mesh wordt hiervan genomen
	cuda.SecondDerivative(dst, u.Buffer(), U.Mesh())

}
