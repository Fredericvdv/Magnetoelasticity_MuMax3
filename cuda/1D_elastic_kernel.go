//package is altijd de folder waar je in zit
package cuda

import (
	"github.com/mumax/3/data"
)


func SecondDerivative(dst, m *data.Slice, mesh *data.Mesh) {
	N := mesh.Size()
	//cfg = structuur van blocks and grids om de cuda te laten lopen
	cfg := make3DConf(N)
	//DevPtr --> geeft specifieke adres van de pointer
	k_SecondDerivative_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z), 
	m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),N[X], N[Y], N[Z], cfg)
}
