package cuda

import (
	"github.com/mumax/3/data"
)

func Poynting(dst, du, normS, shearS *data.Slice, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_poynting_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		du.DevPtr(X), du.DevPtr(Y), du.DevPtr(Z), normS.DevPtr(X), normS.DevPtr(Y), normS.DevPtr(Z),
		shearS.DevPtr(X), shearS.DevPtr(Y), shearS.DevPtr(Z), N[X], N[Y], N[Z], cfg)
}
