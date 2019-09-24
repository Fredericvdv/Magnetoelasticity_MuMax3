package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Classical 4th order RK solver.
type elasYOSH struct{}

func (_ *elasYOSH) Step() {

	//################
	// Differential equation:
	// du/dt = v(t)
	// dv(t)/dt = [f(t) + bf(t) - eta*g(t)]/rho
	// dv(t)/dt = right
	// with f(t) = nabla sigma
	//#################################

	//Initialisation:
	u := U.Buffer()
	size := u.Size()

	//Set fixed displacement
	SetFreezeDisp()
	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)

	v := DU.Buffer()
	v0 := cuda.Buffer(3, size)
	defer cuda.Recycle(v0)
	data.Copy(v0, v)

	a1, a2, a3 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(a1)
	defer cuda.Recycle(a2)
	defer cuda.Recycle(a3)

	//f(t) = nabla sigma
	f := cuda.Buffer(3, size)
	defer cuda.Recycle(f)

	//#############################
	//Time
	if FixDt != 0 {
		Dt_si = FixDt
	}
	dt := float32(Dt_si)
	util.Assert(dt > 0)
	Time += Dt_si

	//#####################
	//Coefficients
	w0 := float32(-math.Pow(2, 1.0/3) / (2 - math.Pow(2, 1.0/3)))
	w1 := float32(1 / (2 - math.Pow(2, 1.0/3)))
	c1 := float32(w1 / 2)
	c2 := float32((w0 + w1) / 2)

	//#####################
	//Integration

	cuda.Madd2(u, u0, v0, 1, c1*dt)
	calcBndry()
	calcRhs(a1, f, v)
	cuda.Madd2(v, v0, a1, 1, w1*dt)
	cuda.Madd2(u, u, v, 1, c2*dt)
	calcBndry()
	calcRhs(a2, f, v)
	cuda.Madd2(v, v, a2, 1, w0*dt)
	cuda.Madd2(u, u, v, 1, c2*dt)
	calcBndry()
	calcRhs(a3, f, v)
	cuda.Madd2(v, v, a3, 1, w1*dt)
	cuda.Madd2(u, u, v, 1, c1*dt)
	calcBndry()
}

func (_ *elasYOSH) Free() {}
