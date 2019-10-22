package cuda

import (
	"github.com/mumax/3/data"
)

func SecondDerivative(dst, u *data.Slice, mesh *data.Mesh, c1, c2, c3 MSlice) {
	N := mesh.Size()
	w := mesh.CellSize()
	wx := float32(1 / w[0])
	wy := float32(1 / w[1])
	wz := float32(1 / w[2])
	cfg := make3DConf(N)
	pbc := mesh.PBC_code()
	// k_Elastodynamic1_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// 	u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// 	c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// 	pbc, cfg)
	// k_Elastodynamic2_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// 	u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// 	c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// 	pbc, cfg)
	// k_Elastodynamic_freebndry_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// 	u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// 	c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// 	pbc, cfg)

	//Vacuum method
	// k_Elastodynamic_2D_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// 	u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// 	c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// 	pbc, cfg)

	//Adjusted differential equation at edges
	k_Elastos_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
		c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
		pbc, cfg)
	//k_Elastodynamic3_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// pbc, cfg)
}
