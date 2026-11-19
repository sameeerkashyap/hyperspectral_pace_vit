#!/bin/sh
set -eu
mkdir -p /runtime
uv venv "$VIRTUAL_ENV" --python /usr/local/bin/python
# Dependencies stay out of the image. Each pod owns its runtime environment.
uv pip install --python "$VIRTUAL_ENV/bin/python" torch==2.5.1 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
uv pip install --python "$VIRTUAL_ENV/bin/python" "/app${PROJECT_EXTRAS:-}" \
    --constraint /app/container/constraints.txt
if [ -n "${EXTRA_REQUIREMENTS:-}" ]; then
    uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$EXTRA_REQUIREMENTS"
fi
exec "$@"
