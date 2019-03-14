//package is altijd de folder waar je in zit
package cuda

import (
	"github.com/mumax/3/data"
)

//Calculate the space dependent part of the second derivative
func SecondDerivative(dst, m *data.Slice, mesh *data.Mesh) {
	N := mesh.Size()
	cellsizeX := mesh.CellSize()
	//cfg = structuur van blocks and grids om de cuda te laten lopen
	cfg := make3DConf(N)
	pbc := mesh.PBC_code()
	Kv := 182e9   //Bulk modulus in [Pa]
	rho := 8.03e3 //Density in [kg/m**3]
	v := Kv / rho //power of Speed of acoustic wave in [m/s]
	c := float32(v / (cellsizeX[0] * cellsizeX[0]))
	//c := float32(10 / (2 * cellsizeX[0])) 	//First derivative

	//DevPtr --> geeft specifieke adres van de pointer
	k_SecondDerivative_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), N[X], N[Y], N[Z], c, pbc, cfg)
}
