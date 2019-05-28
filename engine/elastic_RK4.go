package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Classical 4th order RK solver.
type elasRK4 struct{}

func (_ *elasRK4) Step() {
	fmt.Println("#########################")
	fmt.Println("Start of solver")
	fmt.Println("Number of steps:", NSteps+NUndone)
	fmt.Println("#########################")

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

	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)
	fmt.Println("Max vector norm u0:", cuda.MaxVecNorm(u0))

	v := DU.Buffer()
	fmt.Println("Max vector norm v:", cuda.MaxVecNorm(v))

	ku1, ku2, ku3, ku4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	kv1, kv2, kv3, kv4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(ku1)
	defer cuda.Recycle(ku2)
	defer cuda.Recycle(ku3)
	defer cuda.Recycle(ku4)
	defer cuda.Recycle(kv1)
	defer cuda.Recycle(kv2)
	defer cuda.Recycle(kv3)
	defer cuda.Recycle(kv4)

	//f(t) = nabla sigma
	f := cuda.Buffer(3, size)
	defer cuda.Recycle(f)

	right := cuda.Buffer(3, size)
	defer cuda.Recycle(right)

	//#############################
	//Time
	if FixDt != 0 {
		Dt_si = FixDt
	}
	t0 := Time
	dt := float32(Dt_si)
	util.Assert(dt > 0)
	Time += Dt_si
	fmt.Println("dt = ", Dt_si)

	//#####################
	// du/dt = v(t) ~ ku
	// dv/dt = right(t) ~ kv
	//Stage 1:
	calcSecondDerivDisp(f)
	fmt.Println("Max vector norm f1:", cuda.MaxVecNorm(f))
	calcRightPart(kv1, f, v)

	ku1 = v

	//Stage 2:
	//u = u0*1 + k1*dt/2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(u, u0, ku1, 1, (1./2.)*dt)
	calcSecondDerivDisp(f)
	fmt.Println("Max vector norm f2:", cuda.MaxVecNorm(f))
	//calcRightPart(kv2, f, ku2) is better but this result in internal loop
	calcRightPart(kv2, f, ku1)

	cuda.Madd2(ku2, v, kv2, 1, (1./2.)*dt)

	//Stage 3:
	//u = u0*1 + k2*dt/2
	cuda.Madd2(u, u0, ku2, 1, (1./2.)*dt)
	calcSecondDerivDisp(f)
	calcRightPart(kv3, f, ku2)

	cuda.Madd2(ku3, v, kv3, 1, (1./2.)*dt)

	//Stage 4:
	//u = u0*1 + k3*dt
	Time = t0 + Dt_si
	cuda.Madd2(u, u0, ku3, 1, 1.*dt)
	calcSecondDerivDisp(f)
	calcRightPart(kv4, f, ku3)

	cuda.Madd2(ku4, v, kv3, 1, 1.*dt)

	//###############################
	//Error calculation
	err := cuda.MaxVecDiff(ku1, ku4)
	err2 := cuda.MaxVecDiff(kv1, kv4)

	if err != 0.0 {
		err = err * float64(dt) / cuda.MaxVecNorm(ku4)
	}
	if err2 != 0.0 {
		err2 = err2 * float64(dt) / cuda.MaxVecNorm(kv4)
	}

	//################################
	//Prints
	fmt.Println("Max vector norm ku1:", cuda.MaxVecNorm(ku1))
	fmt.Println("Max vector norm ku2:", cuda.MaxVecNorm(ku2))
	fmt.Println("Max vector norm ku3:", cuda.MaxVecNorm(ku3))
	fmt.Println("Max vector norm ku4:", cuda.MaxVecNorm(ku4))

	//fmt.Println("Max vector norm kv1:", cuda.MaxVecNorm(kv1))
	//fmt.Println("Max vector norm kv4:", cuda.MaxVecNorm(kv4))

	fmt.Println("err = maxVecDiff * dt /MaxVexNorm", err)
	fmt.Println("err2 = maxVecDiff * dt /MaxVexNorm", err2)

	//##########################
	// adjust next time step
	if (err < MaxErr && err2 < MaxErr) || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution
		madd5(u, u0, ku1, ku2, ku3, ku4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		madd5(v, v, kv1, kv2, kv3, kv4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)

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
		Time = t0
		data.Copy(u, u0)
		NUndone++
		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./3.))
		}
	}
}

func (_ *elasRK4) Free() {}
