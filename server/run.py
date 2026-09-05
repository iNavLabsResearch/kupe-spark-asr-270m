#!/usr/bin/env python
"""Path-safe launcher so `server.app` imports regardless of cwd/uvicorn install."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")))
