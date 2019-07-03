package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import(
	"unsafe"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
)

// CUDA handle for madd4 kernel
var madd4_code cu.Function

// Stores the arguments for madd4 kernel invocation
type madd4_args_t struct{
	 arg_dst unsafe.Pointer
	 arg_src1 unsafe.Pointer
	 arg_fac1 float32
	 arg_src2 unsafe.Pointer
	 arg_fac2 float32
	 arg_src3 unsafe.Pointer
	 arg_fac3 float32
	 arg_src4 unsafe.Pointer
	 arg_fac4 float32
	 arg_N int
	 argptr [10]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for madd4 kernel invocation
var madd4_args madd4_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	 madd4_args.argptr[0] = unsafe.Pointer(&madd4_args.arg_dst)
	 madd4_args.argptr[1] = unsafe.Pointer(&madd4_args.arg_src1)
	 madd4_args.argptr[2] = unsafe.Pointer(&madd4_args.arg_fac1)
	 madd4_args.argptr[3] = unsafe.Pointer(&madd4_args.arg_src2)
	 madd4_args.argptr[4] = unsafe.Pointer(&madd4_args.arg_fac2)
	 madd4_args.argptr[5] = unsafe.Pointer(&madd4_args.arg_src3)
	 madd4_args.argptr[6] = unsafe.Pointer(&madd4_args.arg_fac3)
	 madd4_args.argptr[7] = unsafe.Pointer(&madd4_args.arg_src4)
	 madd4_args.argptr[8] = unsafe.Pointer(&madd4_args.arg_fac4)
	 madd4_args.argptr[9] = unsafe.Pointer(&madd4_args.arg_N)
	 }

// Wrapper for madd4 CUDA kernel, asynchronous.
func k_madd4_async ( dst unsafe.Pointer, src1 unsafe.Pointer, fac1 float32, src2 unsafe.Pointer, fac2 float32, src3 unsafe.Pointer, fac3 float32, src4 unsafe.Pointer, fac4 float32, N int,  cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("madd4")
	}

	madd4_args.Lock()
	defer madd4_args.Unlock()

	if madd4_code == 0{
		madd4_code = fatbinLoad(madd4_map, "madd4")
	}

	 madd4_args.arg_dst = dst
	 madd4_args.arg_src1 = src1
	 madd4_args.arg_fac1 = fac1
	 madd4_args.arg_src2 = src2
	 madd4_args.arg_fac2 = fac2
	 madd4_args.arg_src3 = src3
	 madd4_args.arg_fac3 = fac3
	 madd4_args.arg_src4 = src4
	 madd4_args.arg_fac4 = fac4
	 madd4_args.arg_N = N
	

	args := madd4_args.argptr[:]
	cu.LaunchKernel(madd4_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("madd4")
	}
}

// maps compute capability on PTX code for madd4 kernel.
var madd4_map = map[int]string{ 0: "" ,
30: madd4_ptx_30 ,
35: madd4_ptx_35 ,
37: madd4_ptx_37 ,
50: madd4_ptx_50 ,
52: madd4_ptx_52 ,
53: madd4_ptx_53 ,
60: madd4_ptx_60 ,
61: madd4_ptx_61 ,
70: madd4_ptx_70 ,
75: madd4_ptx_75  }

// madd4 PTX code for various compute capabilities.
const(
  madd4_ptx_30 = `
.version 6.4
.target sm_30
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_35 = `
.version 6.4
.target sm_35
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_37 = `
.version 6.4
.target sm_37
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_50 = `
.version 6.4
.target sm_50
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_52 = `
.version 6.4
.target sm_52
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_53 = `
.version 6.4
.target sm_53
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_60 = `
.version 6.4
.target sm_60
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_61 = `
.version 6.4
.target sm_61
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_70 = `
.version 6.4
.target sm_70
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
   madd4_ptx_75 = `
.version 6.4
.target sm_75
.address_size 64

	// .globl	madd4

.visible .entry madd4(
	.param .u64 madd4_param_0,
	.param .u64 madd4_param_1,
	.param .f32 madd4_param_2,
	.param .u64 madd4_param_3,
	.param .f32 madd4_param_4,
	.param .u64 madd4_param_5,
	.param .f32 madd4_param_6,
	.param .u64 madd4_param_7,
	.param .f32 madd4_param_8,
	.param .u32 madd4_param_9
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<13>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd1, [madd4_param_0];
	ld.param.u64 	%rd2, [madd4_param_1];
	ld.param.f32 	%f1, [madd4_param_2];
	ld.param.u64 	%rd3, [madd4_param_3];
	ld.param.f32 	%f2, [madd4_param_4];
	ld.param.u64 	%rd4, [madd4_param_5];
	ld.param.f32 	%f3, [madd4_param_6];
	ld.param.u64 	%rd5, [madd4_param_7];
	ld.param.f32 	%f4, [madd4_param_8];
	ld.param.u32 	%r2, [madd4_param_9];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.f32 	%f5, [%rd8];
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.nc.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f2;
	fma.rn.f32 	%f8, %f5, %f1, %f7;
	cvta.to.global.u64 	%rd11, %rd4;
	add.s64 	%rd12, %rd11, %rd7;
	ld.global.nc.f32 	%f9, [%rd12];
	fma.rn.f32 	%f10, %f9, %f3, %f8;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd7;
	ld.global.nc.f32 	%f11, [%rd14];
	fma.rn.f32 	%f12, %f11, %f4, %f10;
	cvta.to.global.u64 	%rd15, %rd1;
	add.s64 	%rd16, %rd15, %rd7;
	st.global.f32 	[%rd16], %f12;

BB0_2:
	ret;
}


`
 )
