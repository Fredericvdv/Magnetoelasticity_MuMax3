package gpu

import (
	"github.com/barnex/cuda5/cu"
	"nimble-cube/core"
)

// Uploads data from host to GPU.
type Uploader struct {
	host   core.RChan1
	dev    core.Chan1
	bsize  int
	stream cu.Stream
}

func NewUploader(hostdata core.RChan1, devdata core.Chan1) *Uploader {
	core.Assert(hostdata.Size() == devdata.Size())
	blocklen := core.Prod(core.BlockSize(hostdata.Size()))
	return &Uploader{hostdata, devdata, blocklen, 0}
}

func (u *Uploader) Run() {
	core.Debug("run gpu.uploader with block size", u.bsize)
	LockCudaThread()
	defer UnlockCudaThread()
	u.stream = cu.StreamCreate()
	MemHostRegister(u.host.UnsafeData())

	for {
		in := u.host.ReadNext(u.bsize).Host()
		out := u.dev.WriteNext(u.bsize).Device()
		out.CopyHtoDAsync(in, u.stream)
		u.stream.Synchronize()
		u.dev.WriteDone()
		u.host.ReadDone()
	}
}

func RunUploader(tag string, input core.Chan) core.ChanN {
	in := input.ChanN()

	output := core.MakeChanN(in.NComp(), tag, in.Unit(), in.Mesh(), core.GPUMemory)

	for i := range in {
		core.Stack(NewUploader(in[i].NewReader(), output[i]))
	}
	return output
}
