"""
Basic authentication helper for the backend.

`@author`: DAShaikh10
"""

import base64
import secrets
from http import HTTPStatus

from fastapi import HTTPException, Request

# Store credentials in a mutable mapping to avoid reassigning a module-level.
BASIC_AUTH: dict = {}


def set_basic_auth(credentials: tuple) -> None:
    """
    Set BASIC auth credentials (username, password).

    Args:
        credentials: Tuple of (username, password) or `None` to disable auth.
    """

    BASIC_AUTH["creds"] = credentials


async def verify_basic_auth(request: Request) -> bool:
    """
    Validates HTTP Basic Authorization header.

    If no credentials have been configured via :func:`set_basic_auth`, the
    function is a no-op.

    Args:
        request: The incoming HTTP request.

    Returns:
        `True` if authentication is successful or not configured, `False` otherwise.

    Raises:
        HTTPException: If authentication fails.
    """

    auth_header: str | None = request.headers.get("authorization")
    if auth_header:
        try:
            scheme, credentials = auth_header.split(" ", 1)
            if scheme.lower() == "basic":
                decoded_bytes = base64.b64decode(credentials)
                decoded_str = decoded_bytes.decode("utf-8")
                username, password = decoded_str.split(":", 1)

                correct_user = secrets.compare_digest(username, BASIC_AUTH[0])
                correct_pass = secrets.compare_digest(password, BASIC_AUTH[1])

                if correct_user and correct_pass:
                    return True

            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Unauthorized")

        # pylint: disable=broad-except

        except ValueError, Exception:
            pass  # Treat malformed headers as unauthorized.

        # pylint: enable=broad-except

    raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Unauthorized")
