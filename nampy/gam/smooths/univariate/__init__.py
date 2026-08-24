from .cr import CubicSplineTerm
from .ps import PSplineTerm1D
from .tp import ThinPlateSplineTerm

cr = cs = cc = CubicSplineTerm
cp = ps = PSplineTerm1D
tp = ts = ThinPlateSplineTerm

__all__ = ["CubicSplineTerm", "PSplineTerm1D", "ThinPlateSplineTerm"]
__all__ += ["cr", "cs", "cc", "cp", "ps", "tp", "ts"]
