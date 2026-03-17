# Stage 0: Build DSL compiler to WebAssembly for the dashboard frontend
# The WASM target only uses pkg/dsl (pure Go), but src/semantic-router/go.mod
# has local replace directives. Provide stub modules so go mod download succeeds.
FROM --platform=$BUILDPLATFORM golang:1.24 AS wasm-builder
WORKDIR /build
COPY src/semantic-router/ ./
RUN mkdir -p /candle-binding /ml-binding /nlp-binding && \
    echo 'module github.com/vllm-project/semantic-router/candle-binding' > /candle-binding/go.mod && \
    echo 'module github.com/vllm-project/semantic-router/ml-binding' > /ml-binding/go.mod && \
    echo 'module github.com/vllm-project/semantic-router/nlp-binding' > /nlp-binding/go.mod
RUN go mod download
RUN CGO_ENABLED=0 GOOS=js GOARCH=wasm go build \
      -buildvcs=false \
      -ldflags="-s -w" \
      -o /build/signal-compiler.wasm \
      ./cmd/wasm/ && \
    if [ -f "$(go env GOROOT)/lib/wasm/wasm_exec.js" ]; then \
      cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" /build/wasm_exec.js; \
    else \
      cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" /build/wasm_exec.js; \
    fi

# Stage 1: Build frontend with Node.js
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json dashboard/frontend/tsconfig.json dashboard/frontend/tsconfig.node.json dashboard/frontend/vite.config.ts ./
COPY dashboard/frontend/src ./src
COPY dashboard/frontend/public ./public
COPY dashboard/frontend/index.html ./
# Use official npm registry to avoid mirror timeouts
RUN npm config set registry https://registry.npmjs.org/ && \
    npm install --include=dev  # Ensure TypeScript/Vite are installed regardless of NODE_ENV
COPY --from=wasm-builder /build/signal-compiler.wasm /build/wasm_exec.js ./public/
RUN npm run build

# Stage 2: Build backend with Go
FROM golang:1.24 AS backend-builder
WORKDIR /app

# Use Go proxy for reliable downloads
ENV GOPROXY=https://proxy.golang.org,direct

# Accept build arguments for target architecture
ARG TARGETARCH
ARG TARGETOS=linux

# Copy src/semantic-router first (required by dashboard/backend/go.mod replace directive)
# Only copy the packages we actually need (config, modelinventory, observability, startupstatus) - NOT pkg/cli which has extra deps
COPY src/semantic-router/go.mod src/semantic-router/go.sum /app/src/semantic-router/
COPY src/semantic-router/pkg/config/ /app/src/semantic-router/pkg/config/
COPY src/semantic-router/pkg/modelinventory/ /app/src/semantic-router/pkg/modelinventory/
COPY src/semantic-router/pkg/observability/ /app/src/semantic-router/pkg/observability/
COPY src/semantic-router/pkg/startupstatus/ /app/src/semantic-router/pkg/startupstatus/

# Copy go.mod and go.sum first for better caching
COPY dashboard/backend/go.mod dashboard/backend/go.sum /app/dashboard/backend/
WORKDIR /app/dashboard/backend
RUN go mod download

# Copy source code and build
COPY dashboard/backend/ /app/dashboard/backend/
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-w -s" -o /app/dashboard-backend main.go

# Stage 3: Final runtime image
FROM alpine:3.19
RUN apk --no-cache add ca-certificates wget su-exec python3 py3-pip docker-cli
WORKDIR /app
COPY --from=backend-builder /app/dashboard-backend /app/dashboard-backend
COPY --from=frontend-builder /app/frontend/dist /app/frontend
COPY dashboard/backend/config/openclaw-skills.json /app/config/openclaw-skills.json
COPY src/vllm-sr/cli /app/python/cli
# Split-runtime config apply still needs the local Python CLI helpers.
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    'click>=8.1.7' \
    'pyyaml>=6.0.2' \
    'jinja2>=3.1.4' \
    'requests>=2.31.0' \
    'pydantic>=2.0.0'
# Copy entrypoint script
COPY dashboard/backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
RUN addgroup -g 65532 nonroot && \
    adduser -D -u 65532 -G nonroot nonroot && \
    chown -R nonroot:nonroot /app
ENV PYTHONPATH=/app/python
EXPOSE 8700
# Note: We run as root initially so entrypoint can fix ownership/permissions, then switch to nonroot
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/dashboard-backend", "-port=8700", "-static=/app/frontend", "-config=/app/config/config.yaml"]
