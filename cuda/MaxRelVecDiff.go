//package is altijd de folder waar je in zit
package cuda

// import (
// 	"github.com/mumax/3/data"
// )

// //Calculate relative error
// func RelMaxVecDiff(u1, u2 *data.Slice, mesh *data.Mesh) float64 {
// 	N := mesh.Size()
// 	//cfg = structuur van blocks and grids om de cuda te laten lopen
// 	cfg := make3DConf(N)
// 	//out = NewSlice(nComp int, size [3]int)
// 	in := NewSlice(1, N)
// 	var out *float32
// 	k_RelMaxVecDiff_async(out, in.DevPtr(X), u1.DevPtr(X), u1.DevPtr(Y), u1.DevPtr(Z),
// 		u2.DevPtr(X), u2.DevPtr(Y), u2.DevPtr(Z), N[X], N[Y], N[Z], cfg)

// 	//Make it accecible on CPU
// 	//outt := out.HostCopy()

// 	// //Sequential code
// 	// val := 0.0
// 	// max := 0.0
// 	// for ix := 0; ix < N[0]; ix++ {
// 	// 	for iy := 0; iy < N[1]; iy++ {
// 	// 		for iz := 0; iz < N[2]; iz++ {
// 	// 			val = outt.Get(0, ix, iy, iz)
// 	// 			if val > max {
// 	// 				max = val
// 	// 			}
// 	// 		}
// 	// 	}
// 	// }

// 	return &out
// }
