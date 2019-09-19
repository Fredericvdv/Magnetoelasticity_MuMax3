package cuda

import (
	"github.com/mumax/3/data"
)

func Bndry(u *data.Slice, mesh *data.Mesh, c1, c2 float32) {
	N := mesh.Size()
	w := mesh.CellSize()
	wx := float32(1 / w[0])
	wy := float32(1 / w[1])
	cfg := make3DConf(N)
	k_Bndryy_async(u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, 
					c1, c2, cfg)
}
