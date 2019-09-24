package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Add magneto-elasticit coupling field to the effective field.
// see magnetoelasticfield.cu
func AddMagnetoelasticField(Beff, m, norm, shear *data.Slice, B1, B2, Msat MSlice) {
	util.Argument(Beff.Size() == m.Size())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_addmagnetoelasticfield_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		norm.DevPtr(X), norm.DevPtr(Y), norm.DevPtr(Z), shear.DevPtr(X), shear.DevPtr(Y), shear.DevPtr(Z),
		B1.DevPtr(0), B1.Mul(0), B2.DevPtr(0), B2.Mul(0),
		Msat.DevPtr(0), Msat.Mul(0),
		N, cfg)
}

// Calculate magneto-elasticit force density
// see magnetoelasticforce.cu
func GetMagnetoelasticForceDensity(out, m *data.Slice, B1, B2 MSlice, mesh *data.Mesh) {
	util.Argument(out.Size() == m.Size())

	cellsize := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	rcsx := float32(1.0 / cellsize[X])
	rcsy := float32(1.0 / cellsize[Y])
	rcsz := float32(1.0 / cellsize[Z])

	k_getmagnetoelasticforce_async(out.DevPtr(X), out.DevPtr(Y), out.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		B1.DevPtr(0), B1.Mul(0), B2.DevPtr(0), B2.Mul(0),
		rcsx, rcsy, rcsz,
		N[X], N[Y], N[Z],
		mesh.PBC_code(), cfg)
}
