package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var reducemaxabs_code cu.Function

type reducemaxabs_args struct {
	arg_src     unsafe.Pointer
	arg_dst     unsafe.Pointer
	arg_initVal float32
	arg_n       int
	argptr      [4]unsafe.Pointer
}

// Wrapper for reducemaxabs CUDA kernel, asynchronous.
func k_reducemaxabs_async(src unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config, str int) {
	if reducemaxabs_code == 0 {
		reducemaxabs_code = fatbinLoad(reducemaxabs_map, "reducemaxabs")
	}

	var _a_ reducemaxabs_args

	_a_.arg_src = src
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_src)
	_a_.arg_dst = dst
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_dst)
	_a_.arg_initVal = initVal
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_initVal)
	_a_.arg_n = n
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_n)

	args := _a_.argptr[:]
	cu.LaunchKernel(reducemaxabs_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream[str], args)
}

// Wrapper for reducemaxabs CUDA kernel, synchronized.
func k_reducemaxabs(src unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config) {
	const stream = 0
	k_reducemaxabs_async(src, dst, initVal, n, cfg, stream)
	Sync(stream)
}

var reducemaxabs_map = map[int]string{0: "",
	20: reducemaxabs_ptx_20,
	30: reducemaxabs_ptx_30,
	35: reducemaxabs_ptx_35}

const (
	reducemaxabs_ptx_20 = `
.version 3.1
.target sm_20
.address_size 64


.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<40>;
	.reg .f32 	%f<31>;
	.reg .s64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 __cuda_local_var_33800_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd5, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 2 8 1
	mov.u32 	%r39, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r38, %r39, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r39, %r11;
	.loc 2 8 1
	setp.ge.s32 	%p1, %r38, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	.loc 2 8 1
	mul.wide.s32 	%rd6, %r38, 4;
	add.s64 	%rd7, %rd2, %rd6;
	ld.global.f32 	%f5, [%rd7];
	.loc 3 395 5
	abs.f32 	%f6, %f5;
	.loc 3 435 5
	max.f32 	%f30, %f30, %f6;
	.loc 2 8 1
	add.s32 	%r38, %r38, %r4;
	.loc 2 8 1
	setp.lt.s32 	%p2, %r38, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	.loc 2 8 1
	mul.wide.s32 	%rd8, %r2, 4;
	mov.u64 	%rd9, __cuda_local_var_33800_35_non_const_sdata;
	add.s64 	%rd3, %rd9, %rd8;
	st.shared.f32 	[%rd3], %f30;
	bar.sync 	0;
	.loc 2 8 1
	setp.lt.u32 	%p3, %r39, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	.loc 2 8 1
	mov.u32 	%r7, %r39;
	shr.u32 	%r39, %r7, 1;
	.loc 2 8 1
	setp.ge.u32 	%p4, %r2, %r39;
	@%p4 bra 	BB0_5;

	.loc 2 8 1
	ld.shared.f32 	%f7, [%rd3];
	add.s32 	%r15, %r39, %r2;
	mul.wide.u32 	%rd10, %r15, 4;
	add.s64 	%rd12, %rd9, %rd10;
	ld.shared.f32 	%f8, [%rd12];
	.loc 3 435 5
	max.f32 	%f9, %f7, %f8;
	.loc 2 8 1
	st.shared.f32 	[%rd3], %f9;

BB0_5:
	.loc 2 8 1
	bar.sync 	0;
	.loc 2 8 1
	setp.gt.u32 	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	.loc 2 8 1
	setp.gt.s32 	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	.loc 2 8 1
	ld.volatile.shared.f32 	%f10, [%rd3];
	ld.volatile.shared.f32 	%f11, [%rd3+128];
	.loc 3 435 5
	max.f32 	%f12, %f10, %f11;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f12;
	ld.volatile.shared.f32 	%f13, [%rd3+64];
	ld.volatile.shared.f32 	%f14, [%rd3];
	.loc 3 435 5
	max.f32 	%f15, %f14, %f13;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f15;
	ld.volatile.shared.f32 	%f16, [%rd3+32];
	ld.volatile.shared.f32 	%f17, [%rd3];
	.loc 3 435 5
	max.f32 	%f18, %f17, %f16;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f18;
	ld.volatile.shared.f32 	%f19, [%rd3+16];
	ld.volatile.shared.f32 	%f20, [%rd3];
	.loc 3 435 5
	max.f32 	%f21, %f20, %f19;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f21;
	ld.volatile.shared.f32 	%f22, [%rd3+8];
	ld.volatile.shared.f32 	%f23, [%rd3];
	.loc 3 435 5
	max.f32 	%f24, %f23, %f22;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f24;
	ld.volatile.shared.f32 	%f25, [%rd3+4];
	ld.volatile.shared.f32 	%f26, [%rd3];
	.loc 3 435 5
	max.f32 	%f27, %f26, %f25;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f27;

BB0_8:
	.loc 2 8 1
	setp.ne.s32 	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	.loc 2 8 1
	ld.shared.f32 	%f28, [__cuda_local_var_33800_35_non_const_sdata];
	.loc 3 395 5
	abs.f32 	%f29, %f28;
	.loc 2 8 1
	mov.b32 	 %r36, %f29;
	.loc 3 1881 5
	atom.global.max.s32 	%r37, [%rd1], %r36;

BB0_10:
	.loc 2 9 2
	ret;
}


`
	reducemaxabs_ptx_30 = `
.version 3.1
.target sm_30
.address_size 64


.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<40>;
	.reg .f32 	%f<31>;
	.reg .s64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 __cuda_local_var_33873_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd5, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 2 8 1
	mov.u32 	%r39, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r38, %r39, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r39, %r11;
	.loc 2 8 1
	setp.ge.s32 	%p1, %r38, %r9;
	@%p1 bra 	BB0_2;

BB0_1:
	.loc 2 8 1
	mul.wide.s32 	%rd6, %r38, 4;
	add.s64 	%rd7, %rd2, %rd6;
	ld.global.f32 	%f5, [%rd7];
	.loc 3 395 5
	abs.f32 	%f6, %f5;
	.loc 3 435 5
	max.f32 	%f30, %f30, %f6;
	.loc 2 8 1
	add.s32 	%r38, %r38, %r4;
	.loc 2 8 1
	setp.lt.s32 	%p2, %r38, %r9;
	@%p2 bra 	BB0_1;

BB0_2:
	.loc 2 8 1
	mul.wide.s32 	%rd8, %r2, 4;
	mov.u64 	%rd9, __cuda_local_var_33873_35_non_const_sdata;
	add.s64 	%rd3, %rd9, %rd8;
	st.shared.f32 	[%rd3], %f30;
	bar.sync 	0;
	.loc 2 8 1
	setp.lt.u32 	%p3, %r39, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	.loc 2 8 1
	mov.u32 	%r7, %r39;
	shr.u32 	%r39, %r7, 1;
	.loc 2 8 1
	setp.ge.u32 	%p4, %r2, %r39;
	@%p4 bra 	BB0_5;

	.loc 2 8 1
	ld.shared.f32 	%f7, [%rd3];
	add.s32 	%r15, %r39, %r2;
	mul.wide.u32 	%rd10, %r15, 4;
	add.s64 	%rd12, %rd9, %rd10;
	ld.shared.f32 	%f8, [%rd12];
	.loc 3 435 5
	max.f32 	%f9, %f7, %f8;
	.loc 2 8 1
	st.shared.f32 	[%rd3], %f9;

BB0_5:
	.loc 2 8 1
	bar.sync 	0;
	.loc 2 8 1
	setp.gt.u32 	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	.loc 2 8 1
	setp.gt.s32 	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	.loc 2 8 1
	ld.volatile.shared.f32 	%f10, [%rd3];
	ld.volatile.shared.f32 	%f11, [%rd3+128];
	.loc 3 435 5
	max.f32 	%f12, %f10, %f11;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f12;
	ld.volatile.shared.f32 	%f13, [%rd3+64];
	ld.volatile.shared.f32 	%f14, [%rd3];
	.loc 3 435 5
	max.f32 	%f15, %f14, %f13;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f15;
	ld.volatile.shared.f32 	%f16, [%rd3+32];
	ld.volatile.shared.f32 	%f17, [%rd3];
	.loc 3 435 5
	max.f32 	%f18, %f17, %f16;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f18;
	ld.volatile.shared.f32 	%f19, [%rd3+16];
	ld.volatile.shared.f32 	%f20, [%rd3];
	.loc 3 435 5
	max.f32 	%f21, %f20, %f19;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f21;
	ld.volatile.shared.f32 	%f22, [%rd3+8];
	ld.volatile.shared.f32 	%f23, [%rd3];
	.loc 3 435 5
	max.f32 	%f24, %f23, %f22;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f24;
	ld.volatile.shared.f32 	%f25, [%rd3+4];
	ld.volatile.shared.f32 	%f26, [%rd3];
	.loc 3 435 5
	max.f32 	%f27, %f26, %f25;
	.loc 2 8 1
	st.volatile.shared.f32 	[%rd3], %f27;

BB0_8:
	.loc 2 8 1
	setp.ne.s32 	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	.loc 2 8 1
	ld.shared.f32 	%f28, [__cuda_local_var_33873_35_non_const_sdata];
	.loc 3 395 5
	abs.f32 	%f29, %f28;
	.loc 2 8 1
	mov.b32 	 %r36, %f29;
	.loc 3 1881 5
	atom.global.max.s32 	%r37, [%rd1], %r36;

BB0_10:
	.loc 2 9 2
	ret;
}


`
	reducemaxabs_ptx_35 = `
.version 3.1
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 66 3
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 71 3
	ret;
}

.visible .entry reducemaxabs(
	.param .u64 reducemaxabs_param_0,
	.param .u64 reducemaxabs_param_1,
	.param .f32 reducemaxabs_param_2,
	.param .u32 reducemaxabs_param_3
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<40>;
	.reg .f32 	%f<31>;
	.reg .s64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 __cuda_local_var_34022_35_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducemaxabs_param_0];
	ld.param.u64 	%rd5, [reducemaxabs_param_1];
	ld.param.f32 	%f30, [reducemaxabs_param_2];
	ld.param.u32 	%r9, [reducemaxabs_param_3];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 3 8 1
	mov.u32 	%r39, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r38, %r39, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r39, %r11;
	.loc 3 8 1
	setp.ge.s32 	%p1, %r38, %r9;
	@%p1 bra 	BB2_2;

BB2_1:
	.loc 3 8 1
	mul.wide.s32 	%rd6, %r38, 4;
	add.s64 	%rd7, %rd2, %rd6;
	ld.global.f32 	%f5, [%rd7];
	.loc 4 395 5
	abs.f32 	%f6, %f5;
	.loc 4 435 5
	max.f32 	%f30, %f30, %f6;
	.loc 3 8 1
	add.s32 	%r38, %r38, %r4;
	.loc 3 8 1
	setp.lt.s32 	%p2, %r38, %r9;
	@%p2 bra 	BB2_1;

BB2_2:
	.loc 3 8 1
	mul.wide.s32 	%rd8, %r2, 4;
	mov.u64 	%rd9, __cuda_local_var_34022_35_non_const_sdata;
	add.s64 	%rd3, %rd9, %rd8;
	st.shared.f32 	[%rd3], %f30;
	bar.sync 	0;
	.loc 3 8 1
	setp.lt.u32 	%p3, %r39, 66;
	@%p3 bra 	BB2_6;

BB2_3:
	.loc 3 8 1
	mov.u32 	%r7, %r39;
	shr.u32 	%r39, %r7, 1;
	.loc 3 8 1
	setp.ge.u32 	%p4, %r2, %r39;
	@%p4 bra 	BB2_5;

	.loc 3 8 1
	ld.shared.f32 	%f7, [%rd3];
	add.s32 	%r15, %r39, %r2;
	mul.wide.u32 	%rd10, %r15, 4;
	add.s64 	%rd12, %rd9, %rd10;
	ld.shared.f32 	%f8, [%rd12];
	.loc 4 435 5
	max.f32 	%f9, %f7, %f8;
	.loc 3 8 1
	st.shared.f32 	[%rd3], %f9;

BB2_5:
	.loc 3 8 1
	bar.sync 	0;
	.loc 3 8 1
	setp.gt.u32 	%p5, %r7, 131;
	@%p5 bra 	BB2_3;

BB2_6:
	.loc 3 8 1
	setp.gt.s32 	%p6, %r2, 31;
	@%p6 bra 	BB2_8;

	.loc 3 8 1
	ld.volatile.shared.f32 	%f10, [%rd3];
	ld.volatile.shared.f32 	%f11, [%rd3+128];
	.loc 4 435 5
	max.f32 	%f12, %f10, %f11;
	.loc 3 8 1
	st.volatile.shared.f32 	[%rd3], %f12;
	ld.volatile.shared.f32 	%f13, [%rd3+64];
	ld.volatile.shared.f32 	%f14, [%rd3];
	.loc 4 435 5
	max.f32 	%f15, %f14, %f13;
	.loc 3 8 1
	st.volatile.shared.f32 	[%rd3], %f15;
	ld.volatile.shared.f32 	%f16, [%rd3+32];
	ld.volatile.shared.f32 	%f17, [%rd3];
	.loc 4 435 5
	max.f32 	%f18, %f17, %f16;
	.loc 3 8 1
	st.volatile.shared.f32 	[%rd3], %f18;
	ld.volatile.shared.f32 	%f19, [%rd3+16];
	ld.volatile.shared.f32 	%f20, [%rd3];
	.loc 4 435 5
	max.f32 	%f21, %f20, %f19;
	.loc 3 8 1
	st.volatile.shared.f32 	[%rd3], %f21;
	ld.volatile.shared.f32 	%f22, [%rd3+8];
	ld.volatile.shared.f32 	%f23, [%rd3];
	.loc 4 435 5
	max.f32 	%f24, %f23, %f22;
	.loc 3 8 1
	st.volatile.shared.f32 	[%rd3], %f24;
	ld.volatile.shared.f32 	%f25, [%rd3+4];
	ld.volatile.shared.f32 	%f26, [%rd3];
	.loc 4 435 5
	max.f32 	%f27, %f26, %f25;
	.loc 3 8 1
	st.volatile.shared.f32 	[%rd3], %f27;

BB2_8:
	.loc 3 8 1
	setp.ne.s32 	%p7, %r2, 0;
	@%p7 bra 	BB2_10;

	.loc 3 8 1
	ld.shared.f32 	%f28, [__cuda_local_var_34022_35_non_const_sdata];
	.loc 4 395 5
	abs.f32 	%f29, %f28;
	.loc 3 8 1
	mov.b32 	 %r36, %f29;
	.loc 4 1881 5
	atom.global.max.s32 	%r37, [%rd1], %r36;

BB2_10:
	.loc 3 9 2
	ret;
}


`
)
