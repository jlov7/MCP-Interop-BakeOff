FROM node:22-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates python3 python-is-python3 python3-venv \
    && corepack enable pnpm \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/web/package.json apps/web/package.json
RUN uv sync --all-extras --frozen && pnpm install --frozen-lockfile

COPY . .
RUN pnpm --filter @usb-agents/web build

EXPOSE 8765 3000

CMD ["uv", "run", "usb-agents", "serve", "--host", "0.0.0.0", "--port", "8765"]
