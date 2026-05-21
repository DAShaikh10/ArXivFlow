"""
GLiNER powered Label Studio ML backend WSGI.

`@author`: DAShaikh10
"""

import argparse
import os

import uvicorn
from fastapi import FastAPI

from src.api import init_app

from .gliner_inference import LSMLGLiNER

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLiNER Label Studio ML inference server")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9090, help="Server port")
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="Switch debug mode")
    parser.add_argument("--basic-auth-user", default=os.environ.get("BASIC_AUTH_USER", None), help="Basic auth user")
    parser.add_argument("--basic-auth-pass", default=os.environ.get("BASIC_AUTH_PASS", None), help="Basic auth pass")

    args = parser.parse_args()

    app: FastAPI = init_app(LSMLGLiNER, args.basic_auth_user, args.basic_auth_pass)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.debug,
    )
else:
    app: FastAPI = init_app(LSMLGLiNER)
