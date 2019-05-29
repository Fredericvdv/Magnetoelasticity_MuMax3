package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	norm_strain  = NewVectorField("normStrain", "", "Normal strain components", setNormStrain)
	shear_strain = NewVectorField("shearStrain", "", "Shear strain components", setShearStrain)
	Edens_el     = NewScalarField("Edens_el", "J/m3", "Elastic energy density", GetElasticEnergy)
	E_el         = NewScalarValue("E_el", "J", "Elastic energy", GetTotElasticEnergy)
	Edens_kin    = NewScalarField("Edens_kin", "J/m3", "Kinetic energy density", GetKineticEnergy)
	E_kin        = NewScalarValue("E_kin", "J", "Kinetic energy", GetTotKineticEnergy)
)

//###################
//Strain
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

//#############################
//Elastic energy

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

//##################################
// Kinetic energy

func GetKineticEnergy(dst *data.Slice) {
	KineticEnergyDens(dst, DU, Rho)
}

func KineticEnergyDens(dst *data.Slice, du firstDerivative, Rho *RegionwiseScalar) {
	rho, _ := Rho.Slice()
	defer cuda.Recycle(rho)
	cuda.KineticEnergy(dst, du.Buffer(), rho, U.Mesh())
}

func GetTotKineticEnergy() float64 {
	return cellVolume() * float64(cuda.Sum(ValueOf(Edens_kin.Quantity)))
}
