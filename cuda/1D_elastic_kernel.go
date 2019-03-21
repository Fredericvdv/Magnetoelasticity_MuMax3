//package is altijd de folder waar je in zit
package cuda

import (
	"fmt"

	"github.com/mumax/3/data"
)

//Calculate the space dependent part of the second derivative
func SecondDerivative(dst, u *data.Slice, mesh *data.Mesh) {
	N := mesh.Size()
	cellsizeX := mesh.CellSize()
	//cfg = structuur van blocks and grids om de cuda te laten lopen
	cfg := make3DConf(N)
	pbc := mesh.PBC_code()
	fmt.Println("pbc = ", pbc)
	Kv := 182e9   //Bulk modulus in [Pa]
	rho := 8.03e3 //Density in [kg/m**3]
	v := Kv / rho //power of Speed of acoustic wave in [(m/s)^2] --> v' = 4.77 km/s = 4.77 um/ns
	c := v / (cellsizeX[0] * cellsizeX[0])
	//c := v / (2 * cellsizeX[0]) //First derivative
	//c = c * 1e-4 * cellsizeX[0] // Realistic displacement
	cc := float32(c)
	fmt.Println("c = ", c)
	//DevPtr --> geeft specifieke adres van de pointer
	k_SecondDerivative_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], cc, pbc, cfg)
}
