"""TiledArray Python interface — distributed tiled tensor library."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from tiledarray import *  # noqa: F401,F403
