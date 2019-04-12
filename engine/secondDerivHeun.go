package engine

import (
	"fmt"
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
)

// Adaptive Heun solver.
type secondHeun struct{}

// Adaptive Heun method, can be used as solver.Step
func (_ *secondHeun) Step() {
	fmt.Println("#########################")
	fmt.Println("#########################")

	//Set variables of the two first order differential equations:
	//displacement
	y := U.Buffer()
	//First derivative of displacement
	udot := DU.Buffer()
	udot2 := DU.Buffer()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si)
	util.Assert(dt > 0)

	//Set right part of the two first order differential equations:
	//Second derivative of displacement = f(t) = dudot0
	dudot0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dudot0)
	calcSecondDerivDisp(dudot0)
	//First derivative of displacement = g(t) = udot0
	//udot0 := udot
	fmt.Println("average dudot0 = ", sAverageUniverse(dudot0))
	fmt.Println("average y = ", sAverageUniverse(y))
	fmt.Println("average udot = ", sAverageUniverse(udot))

	//Stage 1:
	//y1(t+dt) = y(t) + dt*g(t)
	cuda.Madd2(y, y, udot, 1, dt)
	//g1(t+dt) = g(t) + dt*f(t)
	cuda.Madd2(udot, udot, dudot0, 1, dt)
	//Now, udot = g1(t+dt)

	Time += Dt_si

	// Stage 2:
	// Second derivative of displacement = f(t+dt)
	dudot := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dudot)
	calcSecondDerivDisp(dudot)

	err := cuda.RelMaxVecDiff(dudot,dudot0) * float64(dt)
	err2 := cuda.RelMaxVecDiff(udot, udot2) * float64(dt)
	fmt.Println("err = ", err)
	fmt.Println("MaxErr = ", MaxErr)
	fmt.Println("dt = ", Dt_si)

	// adjust next time step
	if (err < MaxErr && err2<MaxErr) || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// y(t+dt) = y1(t+dt) + 0.5*dt*[g1(t+dt) - g(t)]
		// y(t+dt) = y1(t+dt) + 0.5*dt*[g1(t+dt) - (g1(t+dt)-dt*f(t))]
		// y(t+dt) = y1(t+dt) + 0.5*dt*dt*f(t)
		cuda.Madd2(y, y, dudot0, 1, 0.5*dt*dt)
		// First derivtion of displacement = g(t+dt)= next udot
		// g(t+dt) = g(t) + 0.5*dt*[f(t+dt) + f(t)]
		// g(t+dt) = g1(t+dt) + 0.5*dt*[f(t+dt) - f(t)]
		cuda.Madd3(udot, udot, dudot, dudot0, 1, 0.5*dt, -0.5*dt)
		//Now, udot = g(t+dt)

		//If you run second derivative together with LLG, then remove NSteps++
		NSteps++
		//fmt.Println(MaxErr)
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
		setMaxTorque(dudot)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(udot, udot, dudot0, 1, -dt)
		//Now, udot = g(t)
		cuda.Madd2(y, y, udot, 1, -dt)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (_ *secondHeun) Free() {}
