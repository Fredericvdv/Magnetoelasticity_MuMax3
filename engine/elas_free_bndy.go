package engine

import (
	"github.com/mumax/3/cuda"
)

func calcBndry() {
	Bndry(U, C11, C12)
}

func Bndry(u displacement, C1, C2 *RegionwiseScalar) {
	// c1 := C1.MSlice()
	// defer c1.Recycle()

	// c2 := C2.MSlice()
	// defer c2.Recycle()
	c1:=float32(C1.getRegion(2)[0])
	c2:=float32(C2.getRegion(2)[0])
	cuda.Bndry(u.Buffer(), U.Mesh(), c1, c2)
}
