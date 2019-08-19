package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	norm_stress  = NewVectorField("normStress", "", "Normal stress components", setNormStress)
	shear_stress = NewVectorField("shearStress", "", "Shear stress components", setShearStress)
)

//###################
//Strain
func setNormStress(dst *data.Slice) {
	NormStress(dst, norm_strain.Quantity, C11, C12)
}

func setShearStress(dst *data.Slice) {
	ShearStress(dst, shear_strain.Quantity, C44)
}

func NormStress(dst *data.Slice, eNorm Quantity, C11, C12 *RegionwiseScalar) {
	c1 := C11.MSlice()
	defer c1.Recycle()

	c2 := C12.MSlice()
	defer c2.Recycle()

	enorm := ValueOf(eNorm)
	defer cuda.Recycle(enorm)
	cuda.NormStress(dst, enorm, U.Mesh(), c1, c2)
}

func ShearStress(dst *data.Slice, eShear Quantity, C44 *RegionwiseScalar) {
	c3 := C44.MSlice()
	defer c3.Recycle()

	eshear := ValueOf(eShear)
	defer cuda.Recycle(eshear)
	cuda.ShearStress(dst, eshear, U.Mesh(), c3)
}