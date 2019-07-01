package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Edens_el = NewScalarField("Edens_el", "J/m3", "Elastic energy density", GetElasticEnergy)
	E_el     = NewScalarValue("E_el", "J", "Elastic energy", GetTotElasticEnergy)
)

func GetElasticEnergy(dst *data.Slice) {
	ElasticEnergyDens(dst, ValueOf(norm_strain.Quantity), ValueOf(shear_strain.Quantity), C11, C12, C44)
}

func ElasticEnergyDens(dst, eNorm, eShear *data.Slice, C11, C12, C44 *RegionwiseScalar) {
	c1 := C11.MSlice()
	defer c1.Recycle()

	c2 := C12.MSlice()
	defer c2.Recycle()

	c3 := C44.MSlice()
	defer c3.Recycle()
	cuda.ElasticEnergy(dst, eNorm, eShear, U.Mesh(), c1, c2, c3)
}

func GetTotElasticEnergy() float64 {
	return cellVolume() * float64(cuda.Sum(ValueOf(Edens_el.Quantity)))
}
