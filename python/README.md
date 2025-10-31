# Python Dependencies for Embedding Generation

This directory contains Python dependency specifications for **local development only**.
These are NOT needed for deploying the webapp - the webapp is 100% client-side JavaScript.

## When You Need This

You only need to install these Python dependencies if you are:
- **Building new embeddings** from scratch using the scripts in `scripts/`
- **Developing the core phonolex library** in `src/phonolex/`
- **Running research notebooks** in `research/`

## Installation

```bash
# From project root
pip install -e ./python

# Or with optional dependencies
pip install -e ./python[build]  # For embedding generation
pip install -e ./python[dev]    # For development/testing
```

## Files

- `pyproject.toml` - Modern Python packaging configuration
- `requirements.txt` - Legacy pip requirements (kept for compatibility)

## Webapp Deployment

The webapp at `webapp/frontend/` is a static React app that:
- Loads pre-generated JSON data files from `public/data/`
- Runs entirely in the browser
- Has NO Python backend or dependencies

Netlify ignores this directory and only builds the frontend with Node.js.
