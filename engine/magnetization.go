package engine

type magnetization struct {
	varVectorField
}

var M magnetization // reduced magnetization (unit length)

func init() {
	DeclLValue("m", &M, `Reduced magnetization (unit length)`)
	M.name = "m"
	M.unit = ""
	M.normalized = true
}
