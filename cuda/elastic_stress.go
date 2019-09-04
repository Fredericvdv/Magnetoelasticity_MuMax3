package cuda

import (
	"github.com/mumax/3/data"
)

func NormStress(dst, norm *data.Slice, mesh *data.Mesh, c1, c2 MSlice) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_Normstress_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		norm.DevPtr(X), norm.DevPtr(Y), norm.DevPtr(Z), N[X], N[Y], N[Z], c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), cfg)
}

func ShearStress(dst, shear *data.Slice, mesh *data.Mesh, c3 MSlice) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_Shearstress_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		shear.DevPtr(X), shear.DevPtr(Y), shear.DevPtr(Z), N[X], N[Y], N[Z], c3.DevPtr(0), c3.Mul(0), cfg)
}
