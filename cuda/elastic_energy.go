package cuda

import (
	"github.com/mumax/3/data"
)

func ElasticEnergy(dst, norm, shear *data.Slice, mesh *data.Mesh, c1, c2, c3 MSlice) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_ElsticEnergy_async(dst.DevPtr(X),
		norm.DevPtr(X), norm.DevPtr(Y), norm.DevPtr(Z), shear.DevPtr(X), shear.DevPtr(Y), shear.DevPtr(Z), N[X], N[Y], N[Z],
		c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0), cfg)
}
