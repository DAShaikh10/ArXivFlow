"""
Utility module.

`@author`: DAShaikh10
"""

from .deps import set_basic_auth, verify_basic_auth
from .logger import logger

__all__ = ["logger", "set_basic_auth", "verify_basic_auth"]
