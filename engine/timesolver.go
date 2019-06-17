package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"unsafe"
)

var (
	TheTimeSolver *TimeSolver
)

func init() {
	DeclVar("timesolver", &TheTimeSolver, "The time solver")
	DeclFunc("ConstructTimeSolver", ConstructTimeSolver, "Initialize the time solver")
}

type timeStepper interface {
	Step()
}

type TimeSolver struct {
	timeStepper
	Time          float64
	RightHandSide func(*data.Slice)
	Y             *data.Slice
}

func NewTimeSolver(Y *data.Slice, rhs func(*data.Slice)) *TimeSolver {
	ts := new(TimeSolver)
	ts.timeStepper = &RK4stepper{parent: ts}
	ts.Y = Y
	ts.RightHandSide = rhs
	ts.Time = 0.0
	return ts
}

func (ts *TimeSolver) Run(seconds float64) {
	panic("Run not yet implemented")

}

func (ts *TimeSolver) Steps(nSteps int) {
	Dt_si = FixDt
	for n := 0; n < nSteps; n++ {
		LogOut("step: ", n)
		ts.Step()
	}

}

//------------------------------------

func ConstructTimeSolver() {

	size := M.Mesh().Size()
	nComp := 4

	ptrs := make([]unsafe.Pointer, nComp)
	ptrs[0] = M.Buffer().DevPtr(X)
	ptrs[1] = M.Buffer().DevPtr(Y)
	ptrs[2] = M.Buffer().DevPtr(Z)
	ptrs[3] = cuda.NewSlice(1, size).DevPtr(0)

	Y := data.SliceFromPtrs(size, data.GPUMemory, ptrs)

	TheTimeSolver = NewTimeSolver(Y, rhs)
}

func rhs(dst *data.Slice) {

	size := M.Mesh().Size()
	nComp := 4
	torquePtrs := make([]unsafe.Pointer, nComp)
	torquePtrs[0] = dst.DevPtr(X)
	torquePtrs[1] = dst.DevPtr(Y)
	torquePtrs[2] = dst.DevPtr(Z)
	torque := data.SliceFromPtrs(size, data.GPUMemory, torquePtrs)
	SetTorque(torque)
}
