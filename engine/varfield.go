package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"reflect"
)

// Special buffered quantity to store variable vector fields such as the magnetization
type varVectorField struct {
	buffer_    *data.Slice
	name       string
	unit       string
	normalized bool // if true, normalize after set
}

func (m *varVectorField) Mesh() *data.Mesh    { return Mesh() }
func (m *varVectorField) NComp() int          { return 3 }
func (m *varVectorField) Name() string        { return m.name }
func (m *varVectorField) Unit() string        { return m.unit }
func (m *varVectorField) Buffer() *data.Slice { return m.buffer_ } // todo: rename Gpu()?

func (m *varVectorField) Comp(c int) ScalarField  { return Comp(m, c) }
func (m *varVectorField) SetValue(v interface{})  { m.SetInShape(nil, v.(Config)) }
func (m *varVectorField) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (m *varVectorField) Type() reflect.Type      { return reflect.TypeOf(new(varVectorField)) }
func (m *varVectorField) Eval() interface{}       { return m }
func (m *varVectorField) average() []float64      { return sAverageMagnet(M.Buffer()) }
func (m *varVectorField) Average() data.Vector    { return unslice(m.average()) }
func (m *varVectorField) normalize()              { cuda.Normalize(m.Buffer(), geometry.Gpu()) }

// allocate storage (not done by init, as mesh size may not yet be known then)
func (m *varVectorField) alloc() {
	m.buffer_ = cuda.NewSlice(3, m.Mesh().Size())
	m.Set(RandomMag()) // sane starting config
}

func (b *varVectorField) SetArray(src *data.Slice) {
	if src.Size() != b.Mesh().Size() {
		src = data.Resample(src, b.Mesh().Size())
	}
	data.Copy(b.Buffer(), src)
	if b.normalized {
		b.normalize()
	}
}

func (m *varVectorField) Set(c Config) {
	checkMesh()
	m.SetInShape(nil, c)
}

func (m *varVectorField) LoadFile(fname string) {
	m.SetArray(LoadFile(fname))
}

func (m *varVectorField) Slice() (s *data.Slice, recycle bool) {
	return m.Buffer(), false
}

func (m *varVectorField) EvalTo(dst *data.Slice) {
	data.Copy(dst, m.buffer_)
}

func (m *varVectorField) Region(r int) *vOneReg { return vOneRegion(m, r) }

func (m *varVectorField) String() string { return util.Sprint(m.Buffer().HostCopy()) }

// Set the value of one cell.
func (m *varVectorField) SetCell(ix, iy, iz int, v data.Vector) {
	r := Index2Coord(ix, iy, iz)
	if geometry.shape != nil && !geometry.shape(r[X], r[Y], r[Z]) {
		return
	}
	vNorm := v.Len()
	for c := 0; c < 3; c++ {
		cuda.SetCell(m.Buffer(), c, ix, iy, iz, float32(v[c]/vNorm))
	}
}

// Get the value of one cell.
func (m *varVectorField) GetCell(ix, iy, iz int) data.Vector {
	mx := float64(cuda.GetCell(m.Buffer(), X, ix, iy, iz))
	my := float64(cuda.GetCell(m.Buffer(), Y, ix, iy, iz))
	mz := float64(cuda.GetCell(m.Buffer(), Z, ix, iy, iz))
	return Vector(mx, my, mz)
}

func (m *varVectorField) Quantity() []float64 { return slice(m.Average()) }

// Sets the magnetization inside the shape
func (m *varVectorField) SetInShape(region Shape, conf Config) {
	checkMesh()

	if region == nil {
		region = universe
	}
	host := m.Buffer().HostCopy()
	h := host.Vectors()
	n := m.Mesh().Size()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if region(x, y, z) { // inside
					m := conf(x, y, z)
					h[X][iz][iy][ix] = float32(m[X])
					h[Y][iz][iy][ix] = float32(m[Y])
					h[Z][iz][iy][ix] = float32(m[Z])
				}
			}
		}
	}
	m.SetArray(host)
}

// set m to config in region
func (m *varVectorField) SetRegion(region int, conf Config) {
	host := m.Buffer().HostCopy()
	h := host.Vectors()
	n := m.Mesh().Size()
	r := byte(region)

	regionsArr := regions.HostArray()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				pos := Index2Coord(ix, iy, iz)
				x, y, z := pos[X], pos[Y], pos[Z]
				if regionsArr[iz][iy][ix] == r {
					m := conf(x, y, z)
					h[X][iz][iy][ix] = float32(m[X])
					h[Y][iz][iy][ix] = float32(m[Y])
					h[Z][iz][iy][ix] = float32(m[Z])
				}
			}
		}
	}
	m.SetArray(host)
}

func (m *varVectorField) resize() {
	backup := m.Buffer().HostCopy()
	s2 := Mesh().Size()
	resized := data.Resample(backup, s2)
	m.buffer_.Free()
	m.buffer_ = cuda.NewSlice(VECTOR, s2)
	data.Copy(m.buffer_, resized)
}
