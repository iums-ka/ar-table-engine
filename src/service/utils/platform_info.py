import platform
from enum import Enum, auto

class OS(Enum):
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    UNKNOWN = auto()

_system = platform.system().lower()

if _system == "windows":
    CURRENT_OS = OS.WINDOWS
elif _system == "linux":
    CURRENT_OS = OS.LINUX
elif _system == "darwin":
    CURRENT_OS = OS.MACOS
else:
    CURRENT_OS = OS.UNKNOWN