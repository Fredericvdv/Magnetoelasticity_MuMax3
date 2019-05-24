package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// Sets vector dst to zero where mask != 0.
func CopyMask(dst *data.Slice, mask LUTPtr, vals LUTPtrs, regions *Bytes) {
	N := dst.Len()
	cfg := make1DConf(N)

	for c := 0; c < dst.NComp(); c++ {
		k_copymask_async(dst.DevPtr(c), unsafe.Pointer(mask), unsafe.Pointer(vals[c]), regions.Ptr, N, cfg)
	}
}
