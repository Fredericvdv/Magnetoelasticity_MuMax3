package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Eta = NewScalarParam("eta", "kg/(sm3)", "Damping constant")
	Rho = NewScalarParam("rho", "kg/m3", "Density")
	// 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values
	FrozenDispLoc = NewScalarParam("frozenDispLoc", "", "Defines displacment region that should be fixed")
	FrozenDispVal = NewVectorParam("frozenDispVal", "", "Defines fixed displacement value")
	bf            = NewExcitation("elasticforce", "N/m3", "Defines force density [N/m3]")
)

func FreezeDisp(dst *data.Slice) {
	if !FrozenDispLoc.isZero() {
		Us, _ := U.Slice()
		//defer cuda.Recycle(Us)
		//Set rhs to zero
		cuda.ZeroMask(dst, FrozenDispLoc.gpuLUT1(), regions.Gpu())
		//Set displacment to the given value
		cuda.CopyMask(Us, FrozenDispLoc.gpuLUT1(), FrozenDispVal.gpuLUT(), regions.Gpu())
	}
}

func calcRightPart(dst, f, g *data.Slice) {
	RightPart(dst, f, g, Eta, Rho, bf)
}

func RightPart(dst, f, g *data.Slice, Eta, Rho *RegionwiseScalar, bf *Excitation) {
	//No elastodynamics is calculated if stiffness constants are zero
	if Rho.nonZero() {
		rho, _ := Rho.Slice()
		defer cuda.Recycle(rho)

		eta, _ := Eta.Slice()
		defer cuda.Recycle(eta)

		bf, _ := bf.Slice()
		defer cuda.Recycle(bf)

		cuda.RightPart(dst, f, g, eta, rho, bf)
		//Sufficient to only set right to zero because udot2 = udot+right
		//If initial udot!=0, then do also FreezeDisp(udot2)
		FreezeDisp(dst)
	}
}
