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
	ElasticEnergyDens(dst, norm_strain.Quantity, shear_strain.Quantity, C11, C12, C44)
}

func ElasticEnergyDens(dst *data.Slice, eNorm, eShear Quantity, C11, C12, C44 *RegionwiseScalar) {
	c1 := C11.MSlice()
	defer c1.Recycle()

	c2 := C12.MSlice()
	defer c2.Recycle()

	c3 := C44.MSlice()
	defer c3.Recycle()

	enorm := ValueOf(eNorm)
	defer cuda.Recycle(enorm)

	eshear := ValueOf(eShear)
	defer cuda.Recycle(eshear)

	cuda.ElasticEnergy(dst, enorm, eshear, U.Mesh(), c1, c2, c3)
}

func GetTotElasticEnergy() float64 {
	el_energy := ValueOf(Edens_el.Quantity)
	defer cuda.Recycle(el_energy)
	return cellVolume() * float64(cuda.Sum(el_energy))
}
