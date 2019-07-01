package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Edens_kin = NewScalarField("Edens_kin", "J/m3", "Kinetic energy density", GetKineticEnergy)
	E_kin     = NewScalarValue("E_kin", "J", "Kinetic energy", GetTotKineticEnergy)
)

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
