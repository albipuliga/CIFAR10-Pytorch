# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Inference-first FastAPI web application for CIFAR-10 image classification. Packages two PyTorch CNN models (BaselineCNN and CNNV2) with a web UI, deployed via Docker to Render.

## Commands

```bash
# Install dependencies
uv sync

# Run dev server with hot-reload
uv run uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload

# Build and run Docker container
docker build -t cifar10-fastapi .
docker run --rm -p 8000:8000 cifar10-fastapi

# Check health
curl http://localhost:8000/health

# Interactive API docs
open http://localhost:8000/docs
```

No test suite or lint config exists — testing is manual via the UI or `/docs`.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Server port |
| `MAX_UPLOAD_MB` | `5` | Max image upload size |
| `DEFAULT_MODEL_ID` | `cnnv2` | Model served by default |
| `CHECKPOINTS_DIR` | `src/checkpoints` | Path to `.pth` files |
| `REPORTS_DIR` | `src/reports` | Path to metrics JSON + figures |

## Architecture

**Request flow:**
```
Browser (drag-drop upload) → FastAPI routes → PreprocessService (validate/resize 32×32/normalize)
→ InferenceService (@torch.inference_mode) → ModelRegistry (BaselineCNN/CNNV2 .pth)
→ PredictionResponse (predicted_class, confidence, top_k[], inference_ms, request_id)
```

**Key design decisions:**
- Models are loaded once at startup via lifespan context manager and stored in `app.state`. Startup fails fast if checkpoints are missing.
- All services are accessed via `request.app.state` (dependency injection pattern).
- CPU-only inference (`torch.device("cpu")`) for Render compatibility.
- Structured JSON logging with request IDs and latency via `request_context_middleware`.

## Key Files

| File | Purpose |
|---|---|
| [webapp/main.py](webapp/main.py) | FastAPI app init, lifespan, middleware, static mounts |
| [webapp/api/routes.py](webapp/api/routes.py) | 5 API endpoints: `/`, `/health`, `/api/v1/models`, `/api/v1/predict`, `/api/v1/reports` |
| [webapp/models/cnn.py](webapp/models/cnn.py) | `BaselineCNN` and `CNNV2` nn.Module definitions |
| [webapp/services/model_registry.py](webapp/services/model_registry.py) | Loads `.pth` checkpoints, normalizes state_dict keys |
| [webapp/services/inference.py](webapp/services/inference.py) | Softmax + top-k extraction with latency tracking |
| [webapp/services/preprocess.py](webapp/services/preprocess.py) | MIME validation, size check, resize to 32×32, normalize |
| [webapp/core/config.py](webapp/core/config.py) | `Settings` dataclass, reads env vars, exposes path properties |
| [webapp/core/constants.py](webapp/core/constants.py) | CIFAR-10 class names, checkpoint filenames, normalization params |
| [webapp/schemas/prediction.py](webapp/schemas/prediction.py) | All Pydantic request/response models and `ModelId` enum |
| [src/checkpoints/](src/checkpoints/) | `best_baseline.pth`, `best_cnnv2.pth` — required at runtime |
| [src/reports/](src/reports/) | Optional: `results.json` + figure images served by the reports endpoint |
| [src/cnn.ipynb](src/cnn.ipynb) | Training notebook (not part of runtime) |

## Model Details

- **BaselineCNN:** Conv2d(3→12→24) + FC(24×5×5 → 120 → 84 → 10)
- **CNNV2:** Currently same architecture as BaselineCNN, different checkpoint
- Normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- Input size: 32×32 RGB

## Deployment

Deployed to Render via Docker (`render.yaml`). Push to main → Render auto-deploys. Health check at `/health`.
