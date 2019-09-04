package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for Shearstress kernel
var Shearstress_code cu.Function

// Stores the arguments for Shearstress kernel invocation
type Shearstress_args_t struct {
	arg_sxy    unsafe.Pointer
	arg_syz    unsafe.Pointer
	arg_szx    unsafe.Pointer
	arg_exy    unsafe.Pointer
	arg_eyz    unsafe.Pointer
	arg_ezx    unsafe.Pointer
	arg_Nx     int
	arg_Ny     int
	arg_Nz     int
	arg_C3_    unsafe.Pointer
	arg_C3_mul float32
	argptr     [11]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for Shearstress kernel invocation
var Shearstress_args Shearstress_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	Shearstress_args.argptr[0] = unsafe.Pointer(&Shearstress_args.arg_sxy)
	Shearstress_args.argptr[1] = unsafe.Pointer(&Shearstress_args.arg_syz)
	Shearstress_args.argptr[2] = unsafe.Pointer(&Shearstress_args.arg_szx)
	Shearstress_args.argptr[3] = unsafe.Pointer(&Shearstress_args.arg_exy)
	Shearstress_args.argptr[4] = unsafe.Pointer(&Shearstress_args.arg_eyz)
	Shearstress_args.argptr[5] = unsafe.Pointer(&Shearstress_args.arg_ezx)
	Shearstress_args.argptr[6] = unsafe.Pointer(&Shearstress_args.arg_Nx)
	Shearstress_args.argptr[7] = unsafe.Pointer(&Shearstress_args.arg_Ny)
	Shearstress_args.argptr[8] = unsafe.Pointer(&Shearstress_args.arg_Nz)
	Shearstress_args.argptr[9] = unsafe.Pointer(&Shearstress_args.arg_C3_)
	Shearstress_args.argptr[10] = unsafe.Pointer(&Shearstress_args.arg_C3_mul)
}

// Wrapper for Shearstress CUDA kernel, asynchronous.
func k_Shearstress_async(sxy unsafe.Pointer, syz unsafe.Pointer, szx unsafe.Pointer, exy unsafe.Pointer, eyz unsafe.Pointer, ezx unsafe.Pointer, Nx int, Ny int, Nz int, C3_ unsafe.Pointer, C3_mul float32, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("Shearstress")
	}

	Shearstress_args.Lock()
	defer Shearstress_args.Unlock()

	if Shearstress_code == 0 {
		Shearstress_code = fatbinLoad(Shearstress_map, "Shearstress")
	}

	Shearstress_args.arg_sxy = sxy
	Shearstress_args.arg_syz = syz
	Shearstress_args.arg_szx = szx
	Shearstress_args.arg_exy = exy
	Shearstress_args.arg_eyz = eyz
	Shearstress_args.arg_ezx = ezx
	Shearstress_args.arg_Nx = Nx
	Shearstress_args.arg_Ny = Ny
	Shearstress_args.arg_Nz = Nz
	Shearstress_args.arg_C3_ = C3_
	Shearstress_args.arg_C3_mul = C3_mul

	args := Shearstress_args.argptr[:]
	cu.LaunchKernel(Shearstress_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("Shearstress")
	}
}

// maps compute capability on PTX code for Shearstress kernel.
var Shearstress_map = map[int]string{0: "",
	30: Shearstress_ptx_30,
	35: Shearstress_ptx_35,
	37: Shearstress_ptx_37,
	50: Shearstress_ptx_50,
	52: Shearstress_ptx_52,
	53: Shearstress_ptx_53,
	60: Shearstress_ptx_60,
	61: Shearstress_ptx_61,
	70: Shearstress_ptx_70,
	75: Shearstress_ptx_75}

// Shearstress PTX code for various compute capabilities.
const (
	Shearstress_ptx_30 = `
.version 6.3
.target sm_30
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd3;
	cvta.to.global.u64 	%rd12, %rd6;
	cvta.to.global.u64 	%rd13, %rd2;
	cvta.to.global.u64 	%rd14, %rd5;
	cvta.to.global.u64 	%rd15, %rd1;
	cvta.to.global.u64 	%rd16, %rd4;
	mul.wide.s32 	%rd17, %r4, 4;
	add.s64 	%rd18, %rd16, %rd17;
	ld.global.f32 	%f5, [%rd18];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	add.s64 	%rd19, %rd15, %rd17;
	st.global.f32 	[%rd19], %f7;
	add.s64 	%rd20, %rd14, %rd17;
	ld.global.f32 	%f8, [%rd20];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	add.s64 	%rd21, %rd13, %rd17;
	st.global.f32 	[%rd21], %f10;
	add.s64 	%rd22, %rd12, %rd17;
	ld.global.f32 	%f11, [%rd22];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	add.s64 	%rd23, %rd11, %rd17;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_35 = `
.version 6.3
.target sm_35
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_37 = `
.version 6.3
.target sm_37
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_50 = `
.version 6.3
.target sm_50
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_52 = `
.version 6.3
.target sm_52
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_53 = `
.version 6.3
.target sm_53
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_60 = `
.version 6.3
.target sm_60
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_61 = `
.version 6.3
.target sm_61
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_70 = `
.version 6.3
.target sm_70
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
	Shearstress_ptx_75 = `
.version 6.3
.target sm_75
.address_size 64

	// .globl	Shearstress

.visible .entry Shearstress(
	.param .u64 Shearstress_param_0,
	.param .u64 Shearstress_param_1,
	.param .u64 Shearstress_param_2,
	.param .u64 Shearstress_param_3,
	.param .u64 Shearstress_param_4,
	.param .u64 Shearstress_param_5,
	.param .u32 Shearstress_param_6,
	.param .u32 Shearstress_param_7,
	.param .u32 Shearstress_param_8,
	.param .u64 Shearstress_param_9,
	.param .f32 Shearstress_param_10
)
{
	.reg .pred 	%p<7>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<24>;


	ld.param.u64 	%rd1, [Shearstress_param_0];
	ld.param.u64 	%rd2, [Shearstress_param_1];
	ld.param.u64 	%rd3, [Shearstress_param_2];
	ld.param.u64 	%rd4, [Shearstress_param_3];
	ld.param.u64 	%rd5, [Shearstress_param_4];
	ld.param.u64 	%rd6, [Shearstress_param_5];
	ld.param.u32 	%r5, [Shearstress_param_6];
	ld.param.u32 	%r6, [Shearstress_param_7];
	ld.param.u32 	%r7, [Shearstress_param_8];
	ld.param.u64 	%rd7, [Shearstress_param_9];
	ld.param.f32 	%f14, [Shearstress_param_10];
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r9, %r8, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd8, %rd7;
	mul.wide.s32 	%rd9, %r4, 4;
	add.s64 	%rd10, %rd8, %rd9;
	ld.global.nc.f32 	%f4, [%rd10];
	mul.f32 	%f14, %f4, %f14;

BB0_3:
	cvta.to.global.u64 	%rd11, %rd4;
	mul.wide.s32 	%rd12, %r4, 4;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f5, [%rd13];
	add.f32 	%f6, %f5, %f5;
	mul.f32 	%f7, %f14, %f6;
	cvta.to.global.u64 	%rd14, %rd1;
	add.s64 	%rd15, %rd14, %rd12;
	st.global.f32 	[%rd15], %f7;
	cvta.to.global.u64 	%rd16, %rd5;
	add.s64 	%rd17, %rd16, %rd12;
	ld.global.nc.f32 	%f8, [%rd17];
	add.f32 	%f9, %f8, %f8;
	mul.f32 	%f10, %f14, %f9;
	cvta.to.global.u64 	%rd18, %rd2;
	add.s64 	%rd19, %rd18, %rd12;
	st.global.f32 	[%rd19], %f10;
	cvta.to.global.u64 	%rd20, %rd6;
	add.s64 	%rd21, %rd20, %rd12;
	ld.global.nc.f32 	%f11, [%rd21];
	add.f32 	%f12, %f11, %f11;
	mul.f32 	%f13, %f14, %f12;
	cvta.to.global.u64 	%rd22, %rd3;
	add.s64 	%rd23, %rd22, %rd12;
	st.global.f32 	[%rd23], %f13;

BB0_4:
	ret;
}


`
)
