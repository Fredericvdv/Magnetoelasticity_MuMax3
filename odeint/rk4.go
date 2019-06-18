package odeint

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

type RK4 struct {
	parent *Solver
}

func (s *RK4) Step() {

	Y := s.parent.Y
	t0 := *s.parent.Time
	dt := s.parent.dt
	h := float32(dt)

	nComp := Y.NComp()
	size := Y.Size()

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

	// stage 1
	s.parent.Func(k1)

	// stage 2
	*s.parent.Time = t0 + (1./2.)*dt
	cuda.Madd2(Y, Y, k1, 1, (1./2.)*h) // Y = Y*1 + k1*h/2
	s.parent.ApplyPostChangeFunctions()
	s.parent.Func(k2)

	// stage 3
	cuda.Madd2(Y, Y0, k2, 1, (1./2.)*h) // Y = Y0*1 + k2*1/2
	s.parent.ApplyPostChangeFunctions()
	s.parent.Func(k3)

	// stage 4
	*s.parent.Time = t0 + dt
	cuda.Madd2(Y, Y0, k3, 1, 1.*h) // Y = Y0*1 + k3*1
	s.parent.ApplyPostChangeFunctions()
	s.parent.Func(k4)

	madd5(Y, Y0, k1, k2, k3, k4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
	s.parent.ApplyPostChangeFunctions()
}
