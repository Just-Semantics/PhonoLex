#!/usr/bin/env python
"""
Run the PhonoLex API server.

This script starts the FastAPI server for PhonoLex.
"""

import uvicorn

if __name__ == "__main__":
    print("Starting PhonoLex API server on http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run("phonolex.api.server:app", host="0.0.0.0", port=8000, reload=True) 