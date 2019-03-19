package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"reflect"
)

var DM firstDerivative // firstDerivative (unit A/(ms))

func init() { DeclLValue("dm", &DM, `firstDerivative (unit A/(ms))`) }

// Special buffered quantity to store firstDerivative
// makes sure it's normalized etc.
type firstDerivative struct {
	buffer_ *data.Slice
}

func (dm *firstDerivative) Mesh() *data.Mesh    { return Mesh() }
func (dm *firstDerivative) NComp() int          { return 3 }
func (dm *firstDerivative) Name() string        { return "dm" }
func (dm *firstDerivative) Unit() string        { return "1/s" }
func (dm *firstDerivative) Buffer() *data.Slice { return dm.buffer_ } // todo: rename Gpu()?

func (dm *firstDerivative) Comp(c int) ScalarField  { return Comp(dm, c) }
func (dm *firstDerivative) SetValue(v interface{})  { dm.SetInShape(nil, v.(Config)) }
func (dm *firstDerivative) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (dm *firstDerivative) Type() reflect.Type      { return reflect.TypeOf(new(firstDerivative)) }
func (dm *firstDerivative) Eval() interface{}       { return dm }
func (dm *firstDerivative) average() []float64      { return sAverageMagnet(dm.Buffer()) }
func (dm *firstDerivative) Average() data.Vector    { return unslice(dm.average()) }
func (dm *firstDerivative) normalize()              { cuda.Normalize(dm.Buffer(), geometry.Gpu()) }

// allocate storage (not done by init, as mesh size may not yet be known then)
func (dm *firstDerivative) alloc() {
	dm.buffer_ = cuda.NewSlice(3, dm.Mesh().Size())
	dm.Set(RandomMag()) // sane starting config
}

func (b *firstDerivative) SetArray(src *data.Slice) {
	if src.Size() != b.Mesh().Size() {
		src = data.Resample(src, b.Mesh().Size())
	}
	data.Copy(b.Buffer(), src)
	//b.normalize()
}

func (dm *firstDerivative) Set(c Config) {
	checkMesh()
	dm.SetInShape(nil, c)
}

func (dm *firstDerivative) LoadFile(fname string) {
	dm.SetArray(LoadFile(fname))
}

func (dm *firstDerivative) Slice() (s *data.Slice, recycle bool) {
	return dm.Buffer(), false
}

func (dm *firstDerivative) EvalTo(dst *data.Slice) {
	data.Copy(dst, dm.buffer_)
}

func (dm *firstDerivative) Region(r int) *vOneReg { return vOneRegion(dm, r) }

func (dm *firstDerivative) String() string { return util.Sprint(dm.Buffer().HostCopy()) }

// Set the value of one cell.
func (dm *firstDerivative) SetCell(ix, iy, iz int, v data.Vector) {
	for c := 0; c < 3; c++ {
		cuda.SetCell(dm.Buffer(), c, ix, iy, iz, float32(v[c]))
	}
}

// Get the value of one cell.
func (dm *firstDerivative) GetCell(ix, iy, iz int) data.Vector {
	dmx := float64(cuda.GetCell(dm.Buffer(), X, ix, iy, iz))
	dmy := float64(cuda.GetCell(dm.Buffer(), Y, ix, iy, iz))
	dmz := float64(cuda.GetCell(dm.Buffer(), Z, ix, iy, iz))
	return Vector(dmx, dmy, dmz)
}

func (dm *firstDerivative) Quantity() []float64 { return slice(dm.Average()) }

// Sets the firstDerivative inside the shape
func (dm *firstDerivative) SetInShape(region Shape, conf Config) {
	checkMesh()

	if region == nil {
		region = universe
	}
	host := dm.Buffer().HostCopy()
	h := host.Vectors()
	n := dm.Mesh().Size()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if region(x, y, z) { // inside
					dm := conf(x, y, z)
					h[X][iz][iy][ix] = float32(dm[X])
					h[Y][iz][iy][ix] = float32(dm[Y])
					h[Z][iz][iy][ix] = float32(dm[Z])
				}
			}
		}
	}
	dm.SetArray(host)
}

// set dm to config in region
func (dm *firstDerivative) SetRegion(region int, conf Config) {
	host := dm.Buffer().HostCopy()
	h := host.Vectors()
	n := dm.Mesh().Size()
	r := byte(region)

	regionsArr := regions.HostArray()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				pos := Index2Coord(ix, iy, iz)
				x, y, z := pos[X], pos[Y], pos[Z]
				if regionsArr[iz][iy][ix] == r {
					dm := conf(x, y, z)
					h[X][iz][iy][ix] = float32(dm[X])
					h[Y][iz][iy][ix] = float32(dm[Y])
					h[Z][iz][iy][ix] = float32(dm[Z])
				}
			}
		}
	}
	dm.SetArray(host)
}

func (dm *firstDerivative) resize() {
	backup := dm.Buffer().HostCopy()
	s2 := Mesh().Size()
	resized := data.Resample(backup, s2)
	dm.buffer_.Free()
	dm.buffer_ = cuda.NewSlice(VECTOR, s2)
	data.Copy(dm.buffer_, resized)
}
