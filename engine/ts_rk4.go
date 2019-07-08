package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

type RK4stepper struct {
	parent *TimeSolver
}

func (s *RK4stepper) Step() {

	Y := s.parent.Y
	rhs := s.parent.RightHandSide

	nComp := Y.NComp()
	size := Y.Size()

	t0 := Time

	// backup copy of Y
	Y0 := cuda.Buffer(nComp, size)
	defer cuda.Recycle(Y0)
	data.Copy(Y0, Y)

	k1 := cuda.Buffer(nComp, size)
	k2 := cuda.Buffer(nComp, size)
	k3 := cuda.Buffer(nComp, size)
	k4 := cuda.Buffer(nComp, size)
	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// stage 1
	rhs(k1)

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(Y, Y, k1, 1, (1./2.)*h) // Y = Y*1 + k1*h/2
	M.normalize()
	rhs(k2)

	// stage 3
	cuda.Madd2(Y, Y0, k2, 1, (1./2.)*h) // Y = Y0*1 + k2*1/2
	M.normalize()
	rhs(k3)

	// stage 4
	Time = t0 + Dt_si
	cuda.Madd2(Y, Y0, k3, 1, 1.*h) // Y = Y0*1 + k3*1
	M.normalize()
	rhs(k4)

	madd5(Y, Y0, k1, k2, k3, k4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
}
