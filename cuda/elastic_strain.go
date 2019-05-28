package cuda

import (
	"github.com/mumax/3/data"
)

func NormStrain(dst, u *data.Slice, mesh *data.Mesh, c1 MSlice) {
	N := mesh.Size()
	w := mesh.CellSize()
	wx := float32(1 / w[0])
	wy := float32(1 / w[1])
	wz := float32(1 / w[2])
	cfg := make3DConf(N)
	k_NormStrain_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz, c1.DevPtr(0), c1.Mul(0), cfg)
}

func ShearStrain(dst, u *data.Slice, mesh *data.Mesh, c1 MSlice) {
	N := mesh.Size()
	w := mesh.CellSize()
	wx := float32(1 / w[0])
	wy := float32(1 / w[1])
	wz := float32(1 / w[2])
	cfg := make3DConf(N)
	k_ShearStrain_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz, c1.DevPtr(0), c1.Mul(0), cfg)
}
