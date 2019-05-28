package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	norm_strain  = NewVectorField("normStrain", "", "Normal strain components", setNormStrain)
	shear_strain = NewVectorField("shearStrain", "", "Shear strain components", setShearStrain)
)

func setNormStrain(dst *data.Slice) {
	NormStrain(dst, U, C11)
}

func setShearStrain(dst *data.Slice) {
	ShearStrain(dst, U, C11)
}

func NormStrain(dst *data.Slice, u displacement, C11 *RegionwiseScalar) {
	c1 := C11.MSlice()
	defer c1.Recycle()
	cuda.NormStrain(dst, u.Buffer(), U.Mesh(), c1)
}

func ShearStrain(dst *data.Slice, u displacement, C11 *RegionwiseScalar) {
	c1 := C11.MSlice()
	defer c1.Recycle()
	cuda.ShearStrain(dst, u.Buffer(), U.Mesh(), c1)
}
