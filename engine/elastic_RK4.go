package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Classical 4th order RK solver.
type elasRK4 struct{}

func (_ *elasRK4) Step() {

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

	//#####################
	// du/dt = v(t) ~ ku
	// dv/dt = right(t) ~ kv
	//Stage 1:
	calcRhs(kv1, f, v)

	ku1 = v0

	//Stage 2:
	//u = u0*1 + k1*dt/2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(u, u0, ku1, 1, (1./2.)*dt)
	cuda.Madd2(v, v0, kv1, 1, (1./2.)*dt)
	calcBndry()
	calcRhs(kv2, f, v)

	cuda.Madd2(ku2, v0, kv1, 1, (1./2.)*dt)

	//Stage 3:
	//u = u0*1 + k2*dt/2
	cuda.Madd2(u, u0, ku2, 1, (1./2.)*dt)
	cuda.Madd2(v, v0, kv2, 1, (1./2.)*dt)
	calcBndry()
	calcRhs(kv3, f, v)

	cuda.Madd2(ku3, v0, kv2, 1, (1./2.)*dt)

	//Stage 4:
	//u = u0*1 + k3*dt
	Time = t0 + Dt_si
	cuda.Madd2(u, u0, ku3, 1, 1.*dt)
	cuda.Madd2(v, v0, kv3, 1, 1.*dt)
	calcBndry()
	calcRhs(kv4, f, v)

	cuda.Madd2(ku4, v0, kv3, 1, 1.*dt)

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

	// //################################
	// //Prints
	// fmt.Println("Max vector norm ku1:", cuda.MaxVecNorm(ku1))
	// fmt.Println("Max vector norm ku2:", cuda.MaxVecNorm(ku2))
	// fmt.Println("Max vector norm ku3:", cuda.MaxVecNorm(ku3))
	// fmt.Println("Max vector norm ku4:", cuda.MaxVecNorm(ku4))

	// //fmt.Println("Max vector norm kv1:", cuda.MaxVecNorm(kv1))
	// //fmt.Println("Max vector norm kv4:", cuda.MaxVecNorm(kv4))

	// fmt.Println("err = maxVecDiff * dt /MaxVexNorm", err)
	// fmt.Println("err2 = maxVecDiff * dt /MaxVexNorm", err2)

	//##########################
	// adjust next time step
	if (err < MaxErr && err2 < MaxErr) || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution

		madd5(u, u0, ku1, ku2, ku3, ku4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		madd5(v, v0, kv1, kv2, kv3, kv4, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
		calcBndry()
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
		data.Copy(v, v0)
		NUndone++
		if err > err2 {
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		} else {
			adaptDt(math.Pow(MaxErr/err2, 1./3.))
		}
	}
}

func (_ *elasRK4) Free() {}
