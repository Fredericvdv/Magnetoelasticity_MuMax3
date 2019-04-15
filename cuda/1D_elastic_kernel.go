//package is altijd de folder waar je in zit
package cuda

import (
	"fmt"

	"github.com/mumax/3/data"
)

//Calculate the space dependent part of the second derivative
func SecondDerivative(dst, u *data.Slice, mesh *data.Mesh, c1, c2, c3 MSlice) {
	N := mesh.Size()
	w := mesh.CellSize()
	wx := float32(1 / w[0])
	wy := float32(1 / w[1])
	wz := float32(1 / w[2])
	//cfg = structuur van blocks and grids om de cuda te laten lopen
	cfg := make3DConf(N)
	pbc := mesh.PBC_code()
	fmt.Println("This is fred, 3.8")
	//DevPtr --> geeft specifieke adres van de pointer
	k_SecondDerivative_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
		c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
		pbc, cfg)
}
