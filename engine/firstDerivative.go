package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"reflect"
)

var DU firstDerivative // firstDerivative (unit [m/s])

func init() { DeclLValue("du", &DU, `firstDerivative (unit [m/s])`) }

// Special buffered quantity to store firstDerivative
type firstDerivative struct {
	buffer_ *data.Slice
}

func (du *firstDerivative) Mesh() *data.Mesh    { return Mesh() }
func (du *firstDerivative) NComp() int          { return 3 }
func (du *firstDerivative) Name() string        { return "du/dt" }
func (du *firstDerivative) Unit() string        { return "m/s" }
func (du *firstDerivative) Buffer() *data.Slice { return du.buffer_ } // todo: rename Gpu()?

func (du *firstDerivative) Comp(c int) ScalarField  { return Comp(du, c) }
func (du *firstDerivative) SetValue(v interface{})  { du.SetInShape(nil, v.(Config)) }
func (du *firstDerivative) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (du *firstDerivative) Type() reflect.Type      { return reflect.TypeOf(new(firstDerivative)) }
func (du *firstDerivative) Eval() interface{}       { return du }
func (du *firstDerivative) average() []float64      { return sAverageMagnet(du.Buffer()) }
func (du *firstDerivative) Average() data.Vector    { return unslice(du.average()) }

//func (du *firstDerivative) normalize()              { cuda.Normalize(du.Buffer(), geometry.Gpu()) }

// allocate storage (not done by init, as mesh size may not yet be known then)
func (du *firstDerivative) alloc() {
	du.buffer_ = cuda.NewSlice(3, du.Mesh().Size())
	du.Set(Uniform(0, 0, 0)) // sane starting config
}

func (b *firstDerivative) SetArray(src *data.Slice) {
	if src.Size() != b.Mesh().Size() {
		src = data.Resample(src, b.Mesh().Size())
	}
	data.Copy(b.Buffer(), src)
	//b.normalize()
}

func (du *firstDerivative) Set(c Config) {
	checkMesh()
	du.SetInShape(nil, c)
}

func (du *firstDerivative) LoadFile(fname string) {
	du.SetArray(LoadFile(fname))
}

func (du *firstDerivative) Slice() (s *data.Slice, recycle bool) {
	return du.Buffer(), false
}

func (du *firstDerivative) EvalTo(dst *data.Slice) {
	data.Copy(dst, du.buffer_)
}

func (du *firstDerivative) Region(r int) *vOneReg { return vOneRegion(du, r) }

func (du *firstDerivative) String() string { return util.Sprint(du.Buffer().HostCopy()) }

// Set the value of one cell.
func (du *firstDerivative) SetCell(ix, iy, iz int, v data.Vector) {
	for c := 0; c < 3; c++ {
		cuda.SetCell(du.Buffer(), c, ix, iy, iz, float32(v[c]))
	}
}

// Get the value of one cell.
func (du *firstDerivative) GetCell(ix, iy, iz int) data.Vector {
	dux := float64(cuda.GetCell(du.Buffer(), X, ix, iy, iz))
	duy := float64(cuda.GetCell(du.Buffer(), Y, ix, iy, iz))
	duz := float64(cuda.GetCell(du.Buffer(), Z, ix, iy, iz))
	return Vector(dux, duy, duz)
}

func (du *firstDerivative) Quantity() []float64 { return slice(du.Average()) }

// Sets the firstDerivative inside the shape
func (du *firstDerivative) SetInShape(region Shape, conf Config) {
	checkMesh()

	if region == nil {
		region = universe
	}
	host := du.Buffer().HostCopy()
	h := host.Vectors()
	n := du.Mesh().Size()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if region(x, y, z) { // inside
					du := conf(x, y, z)
					h[X][iz][iy][ix] = float32(du[X])
					h[Y][iz][iy][ix] = float32(du[Y])
					h[Z][iz][iy][ix] = float32(du[Z])
				}
			}
		}
	}
	du.SetArray(host)
}

// set du to config in region
func (du *firstDerivative) SetRegion(region int, conf Config) {
	host := du.Buffer().HostCopy()
	h := host.Vectors()
	n := du.Mesh().Size()
	r := byte(region)

	regionsArr := regions.HostArray()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				pos := Index2Coord(ix, iy, iz)
				x, y, z := pos[X], pos[Y], pos[Z]
				if regionsArr[iz][iy][ix] == r {
					du := conf(x, y, z)
					h[X][iz][iy][ix] = float32(du[X])
					h[Y][iz][iy][ix] = float32(du[Y])
					h[Z][iz][iy][ix] = float32(du[Z])
				}
			}
		}
	}
	du.SetArray(host)
}

func (du *firstDerivative) resize() {
	backup := du.Buffer().HostCopy()
	s2 := Mesh().Size()
	resized := data.Resample(backup, s2)
	du.buffer_.Free()
	du.buffer_ = cuda.NewSlice(VECTOR, s2)
	data.Copy(du.buffer_, resized)
}
