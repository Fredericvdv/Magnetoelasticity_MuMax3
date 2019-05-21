package cuda

import (
	"github.com/mumax/3/data"
)

//Calculate [f(t)-eta*g(t)]/rho
func RightPart(dst, f, g, eta, rho *data.Slice) {

	N := dst.Len()
	cfg := make1DConf(N)

	k_mul_async(dst.DevPtr(0), eta.DevPtr(0), g.DevPtr(0), N, cfg)
	k_mul_async(dst.DevPtr(1), eta.DevPtr(0), g.DevPtr(1), N, cfg)
	k_mul_async(dst.DevPtr(2), eta.DevPtr(0), g.DevPtr(2), N, cfg)

	Madd2(dst, f, dst, 1, -1)

	k_pointwise_div_async(dst.DevPtr(0), dst.DevPtr(0), rho.DevPtr(0), N, cfg)
	k_pointwise_div_async(dst.DevPtr(1), dst.DevPtr(1), rho.DevPtr(0), N, cfg)
	k_pointwise_div_async(dst.DevPtr(2), dst.DevPtr(2), rho.DevPtr(0), N, cfg)
}