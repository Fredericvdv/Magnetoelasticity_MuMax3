package odeint

import (
	"github.com/mumax/3/data"
	"unsafe"
)

type stepper interface {
	Step()
}

// Solves dY/dt = Func(Y,t)
type Solver struct {
	stepper

	Time   *float64
	dt     float64
	inject chan func()
	pause  bool
	nsteps int

	Y          *data.Slice
	funcs      []func(*data.Slice)
	vars       []*data.Slice
	postChange []func()
}

func NewSolver(time *float64, fixdt float64) *Solver {
	s := new(Solver)
	s.stepper = &RK4{parent: s}
	s.Time = time
	s.dt = fixdt
	return s
}

func (s *Solver) SetInjectChannel(inject chan func()) {
	s.inject = inject
}

// Add an ordinary differential equation to solver
func (s *Solver) AddODE(y *data.Slice, f func(*data.Slice)) {
	s.vars = append(s.vars, y)
	s.funcs = append(s.funcs, f)
	s.constructY()
}

// Add a function which is evaluated each time Y changes
func (s *Solver) AddPostChangeFunction(f func()) {
	s.postChange = append(s.postChange, f)
}

// Construcy the Solver.Y slice which containes al the variables
func (s *Solver) constructY() {
	var ptrs []unsafe.Pointer
	size := s.vars[0].Size()
	// TODO: check if all sizes are the same (maybe in AddODE)
	for _, v := range s.vars {
		ptrs = append(ptrs, v.Ptrs()...)
	}
	s.Y = data.SliceFromPtrs(size, data.GPUMemory, ptrs)
}

func (s *Solver) Func(dst *data.Slice) {
	ptrIdx := 0
	for iv, v := range s.vars {
		dst_ := dst.SubSlice(ptrIdx, ptrIdx+v.NComp())
		s.funcs[iv](dst_)
		ptrIdx += v.NComp()
	}
}

func (s *Solver) RunWhile(condition func() bool) {
	// TODO: sanity check
	s.pause = false // may be set by <-Inject
	const output = true
	s.runWhile(condition, output)
	s.pause = true
}

func (s *Solver) runWhile(condition func() bool, output bool) {
	//DoOutput() // allow t=0 output   // TODO
	for condition() && !s.pause {
		select {
		default:
			s.step(output)
		// accept tasks form Inject channel
		case f := <-s.inject:
			f()
		}
	}
}

func (s *Solver) step(output bool) {
	s.stepper.Step()
	s.nsteps++
}

func (s *Solver) Steps(n int) {
	stop := s.nsteps + n
	s.RunWhile(func() bool { return s.nsteps < stop })
}

func (s *Solver) ApplyPostChangeFunctions() {
	for _, f := range s.postChange {
		f()
	}
}
