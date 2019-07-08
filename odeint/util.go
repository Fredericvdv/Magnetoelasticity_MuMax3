package odeint

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// TODO: copied from engine.madd5 -> DRY!  move to cuda package?
func madd5(dst, src1, src2, src3, src4, src5 *data.Slice, w1, w2, w3, w4, w5 float32) {
	cuda.Madd3(dst, src1, src2, src3, w1, w2, w3)
	cuda.Madd3(dst, dst, src4, src5, 1, w4, w5)
}

// TODO: copied from engine.madd6 -> DRY!  move to cuda package?
func madd6(dst, src1, src2, src3, src4, src5, src6 *data.Slice, w1, w2, w3, w4, w5, w6 float32) {
	madd5(dst, src1, src2, src3, src4, src5, w1, w2, w3, w4, w5)
	cuda.Madd2(dst, dst, src6, 1, w6)
}
