package cuda

import (
	"github.com/mumax/3/data"
)

func KineticEnergy(dst, du, rho *data.Slice, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_KineticEnergy_async(dst.DevPtr(X),
		du.DevPtr(X), du.DevPtr(Y), du.DevPtr(Z), rho.DevPtr(0), N[X], N[Y], N[Z], cfg)
}
