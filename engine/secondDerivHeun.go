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
	fmt.Println("Start of solver")
	fmt.Println("Number of steps:", NSteps+NUndone)
	fmt.Println("#########################")

	//################
	// Differential equation:
	// du/dt = g(t)
	// dg(t)/dt = [f(t) - eta*g(t)]/rho
	// dg(t)/dt = right
	// with f(t) = nabla sigma

	//#################################
	//Set initial states and initialisations:
	//displacement
	y := U.Buffer()
	fmt.Println("Max vector norm y:", cuda.MaxVecNorm(y))

	//First derivative of displacement du/dt = g(t) = udot
	udot := DU.Buffer()
	fmt.Println("Max vector norm udot:", cuda.MaxVecNorm(udot))

	//Necessary to calculate error
	udot2 := cuda.Buffer(VECTOR, udot.Size())
	defer cuda.Recycle(udot2)

	//f(t) = nabla sigma = dudot0
	dudot0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dudot0)
	calcSecondDerivDisp(dudot0)

	//f(t+dt)
	//dudot := cuda.Buffer(3, y.Size())
	//defer cuda.Recycle(dudot)

	right := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(right)
	calcRightPart(right, dudot0, udot)

	right2 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(right2)

	//#############################
	//Time
	if FixDt != 0 {
		Dt_si = FixDt
	}
	dt := float32(Dt_si)
	util.Assert(dt > 0)
	Time += Dt_si
	fmt.Println("dt = ", Dt_si)

	//#####################
	//Stage 1: predictor
	//y1(t+dt) = y(t) + dt*g(t)
	cuda.Madd2(y, y, udot, 1, dt)
	calcSecondDerivDisp(dudot0)

	//Without damping:
	//g1(t+dt) = g(t) + dt*f(t)
	// cuda.Madd2(udot2, udot, dudot0, 1, dt)
	//With damping
	//With damping: g1(t+dt) = g(t) + dt*[f(t)-n*g(t)]/rho
	//With damping: g1(t+dt) = g(t) + dt*right
	cuda.Madd2(udot2, udot, right, 1, dt)
	calcRightPart(right2, dudot0, udot2)

	//###############################
	//Error calculation
	err := cuda.MaxVecDiff(right, right2)
	err2 := cuda.MaxVecDiff(udot, udot2)

	if err != 0.0 {
		err = err * float64(dt) / cuda.MaxVecNorm(right2)
	}
	if err2 != 0.0 {
		err2 = err2 * float64(dt) / cuda.MaxVecNorm(udot2)
	}

	//################################
	//Prints
	fmt.Println("Max vector norm y2:", cuda.MaxVecNorm(y))

	fmt.Println("Max vector norm dudot0", cuda.MaxVecNorm(dudot0))
	//fmt.Println("Max vector norm dudot:", cuda.MaxVecNorm(dudot))
	//fmt.Println("Max vector diff dudot & dudot0:", cuda.MaxVecDiff(dudot, dudot0))

	fmt.Println("Max vector norm udot:", cuda.MaxVecNorm(udot))
	fmt.Println("Max vector norm udot2", cuda.MaxVecNorm(udot2))
	fmt.Println("Max vector diff udot & udot2:", cuda.MaxVecDiff(udot, udot2))

	fmt.Println("err = maxVecDiff * dt /MaxVexNorm", err)
	fmt.Println("err2 = maxVecDiff * dt /MaxVexNorm", err2)

	//##########################
	// adjust next time step
	if (err < MaxErr && err2 < MaxErr) || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// y(t+dt) = y1(t+dt) + 0.5*dt*[g1(t+dt) - g(t)]
		// y(t+dt) = y1(t+dt) + 0.5*dt*[g1(t+dt) - (g1(t+dt)-dt*f(t))]
		// y(t+dt) = y1(t+dt) + 0.5*dt*dt*f(t)
		cuda.Madd3(y, y, udot2, udot, 1, dt/2, -dt/2)

		// First derivtion of displacement = g(t+dt)= next udot
		// g(t+dt) = g(t) + 0.5*dt*[f(t+dt) + f(t)]
		// g(t+dt) = g1(t+dt) + 0.5*dt*[f(t+dt) - f(t)]
		//cuda.Madd3(udot, udot2, dudot, dudot0, 1, 0.5*dt, -0.5*dt)

		//TO DO: with damping:
		// g(t+dt) = g(t) + 0.5*dt*[f(t+dt) + f(t)]
		// g(t+dt) = g1(t+dt) + 0.5*dt*[f(t+dt) - f(t)]
		cuda.Madd3(udot, udot, right, right2, 1, dt/2, dt/2)

		//If you run second derivative together with LLG, then remove NSteps++
		NSteps++

		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./2.))
			setLastErr(err)
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./2.))
			setLastErr(err2)
		}

	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, udot, 1, -dt)
		NUndone++
		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./3.))
		}
	}
}

func (_ *secondHeun) Free() {}
