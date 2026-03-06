package router

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
)

func newOpenClawHandler(cfg *config.Config) *handlers.OpenClawHandler {
	if !cfg.OpenClawEnabled {
		return nil
	}

	openClawHandler := handlers.NewOpenClawHandler(cfg.OpenClawDataDir, cfg.ReadonlyMode)
	openClawHandler.SetRouterConfigPath(cfg.AbsConfigPath)

	// Initialize Matrix client for OpenClaw if Matrix is enabled
	// This allows dynamic registration of worker agent users
	if matrixEnabled := os.Getenv("MATRIX_ENABLED"); matrixEnabled == "true" {
		initializeMatrixForOpenClaw(openClawHandler)
	}

	return openClawHandler
}

func initializeMatrixForOpenClaw(openClawHandler *handlers.OpenClawHandler) {
	matrixURL := strings.TrimSpace(os.Getenv("MATRIX_INTERNAL_URL"))
	matrixDomain := strings.TrimSpace(os.Getenv("MATRIX_DOMAIN"))
	matrixRegToken := strings.TrimSpace(os.Getenv("MATRIX_REG_TOKEN"))
	matrixSystemUser := strings.TrimSpace(os.Getenv("MATRIX_SYSTEM_USER"))
	matrixSystemAccessToken := strings.TrimSpace(os.Getenv("MATRIX_SYSTEM_ACCESS_TOKEN"))

	if matrixSystemUser == "" {
		matrixSystemUser = "system"
	}
	if matrixDomain == "" {
		matrixDomain = "matrix.vllm-sr.local"
	}

	if matrixURL == "" || (matrixRegToken == "" && matrixSystemAccessToken == "") {
		log.Printf("Warning: MATRIX_ENABLED=true but MATRIX_INTERNAL_URL or (MATRIX_REG_TOKEN/MATRIX_SYSTEM_ACCESS_TOKEN) not set (worker agents will not have Matrix access)")
		return
	}

	matrixClient, err := handlers.NewMatrixClient(handlers.MatrixClientConfig{
		HomeserverURL:     matrixURL,
		Domain:            matrixDomain,
		SystemUser:        matrixSystemUser,
		RegToken:          matrixRegToken,
		SystemAccessToken: matrixSystemAccessToken,
	})
	if err != nil {
		log.Printf("Warning: failed to initialize Matrix client for OpenClaw: %v (worker agents will not have Matrix access)", err)
		return
	}

	openClawHandler.SetMatrixClient(matrixClient, matrixDomain)
	authMethod := "password"
	if matrixSystemAccessToken != "" {
		authMethod = "access_token"
	}
	log.Printf("Matrix client initialized for OpenClaw worker registration (homeserver=%s, domain=%s, auth=%s)", matrixURL, matrixDomain, authMethod)

	// Initialize MatrixBridge for Matrix-only communication mode
	// NOTE: 已移除 hybrid 模式，所有通信强制走 Matrix
	matrixBridge, bridgeErr := handlers.NewMatrixBridge(handlers.MatrixBridgeConfig{
		Mode:              handlers.ModeMatrix, // 强制 Matrix 模式，无降级
		ServerDomain:      matrixDomain,
		InternalURL:       matrixURL,
		RegToken:          matrixRegToken,
		SystemUser:        matrixSystemUser,
		SystemAccessToken: matrixSystemAccessToken,
		DedupTTL:          5 * time.Minute,
	})
	if bridgeErr != nil {
		log.Printf("Warning: failed to initialize MatrixBridge: %v (Matrix communication disabled - this is now a fatal condition)", bridgeErr)
		return
	}

	openClawHandler.SetMatrixBridge(matrixBridge)
	log.Printf("MatrixBridge initialized for Matrix-only communication (mode=matrix, no fallback)")

	// Start MatrixSyncWorker to sync messages from Matrix to WebSocket clients
	// 使用 NewMatrixSyncWorkerWithHandler 以便能够广播到 WebSocket/SSE 客户端
	syncWorker := handlers.NewMatrixSyncWorkerWithHandler(matrixClient, matrixBridge, openClawHandler)
	go syncWorker.Start(context.Background())
	log.Printf("MatrixSyncWorker started for real-time message synchronization")
}

func registerOpenClawRoutes(
	mux *http.ServeMux,
	cfg *config.Config,
	openClawHandler *handlers.OpenClawHandler,
) {
	if cfg.OpenClawEnabled && openClawHandler != nil {
		registerEnabledOpenClawRoutes(mux, openClawHandler)
		log.Printf("OpenClaw API endpoints registered: /api/openclaw/*")
		registerOpenClawProxyRoute(mux, openClawHandler)
		log.Printf("OpenClaw dynamic proxy configured: /embedded/openclaw/{name}/ (WebSocket enabled)")
		return
	}

	registerDisabledOpenClawRoutes(mux)
	log.Printf("OpenClaw feature disabled")
}

func registerEnabledOpenClawRoutes(mux *http.ServeMux, openClawHandler *handlers.OpenClawHandler) {
	mux.HandleFunc("/api/openclaw/status", openClawHandler.StatusHandler())
	mux.HandleFunc("/api/openclaw/skills", openClawHandler.SkillsHandler())
	mux.HandleFunc("/api/openclaw/teams", openClawHandler.TeamsHandler())
	mux.HandleFunc("/api/openclaw/teams/", openClawHandler.TeamByIDHandler())
	mux.HandleFunc("/api/openclaw/workers", openClawHandler.WorkersHandler())
	mux.HandleFunc("/api/openclaw/workers/", openClawHandler.WorkerByIDHandler())
	mux.HandleFunc("/api/openclaw/rooms", openClawHandler.RoomsHandler())
	mux.HandleFunc("/api/openclaw/rooms/", openClawHandler.RoomByIDHandler())
	mux.HandleFunc("/api/openclaw/provision", openClawHandler.ProvisionHandler())
	mux.HandleFunc("/api/openclaw/start", openClawHandler.StartHandler())
	mux.HandleFunc("/api/openclaw/stop", openClawHandler.StopHandler())
	mux.HandleFunc("/api/openclaw/token", openClawHandler.TokenHandler())
	mux.HandleFunc("/api/openclaw/next-port", openClawHandler.NextPortHandler())
	mux.HandleFunc("/api/openclaw/containers/", openClawHandler.DeleteHandler())
}

func registerOpenClawProxyRoute(mux *http.ServeMux, openClawHandler *handlers.OpenClawHandler) {
	var proxyCache sync.Map // map[string]http.Handler
	mux.HandleFunc("/embedded/openclaw/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		rest := strings.TrimPrefix(r.URL.Path, "/embedded/openclaw/")
		parts := strings.SplitN(rest, "/", 2)
		name := parts[0]
		if name == "" {
			http.Error(w, "container name required in path", http.StatusBadRequest)
			return
		}

		targetBase, ok := openClawHandler.TargetBaseForContainer(name)
		if !ok {
			http.Error(w, "container not found in registry", http.StatusNotFound)
			return
		}

		token := strings.TrimSpace(openClawHandler.GatewayTokenForContainer(name))
		staticHeaders := map[string]string{}
		if token != "" {
			staticHeaders["Authorization"] = "Bearer " + token
			staticHeaders["X-OpenClaw-Token"] = token
		}

		stripPrefix := "/embedded/openclaw/" + name
		cacheKey := fmt.Sprintf("%s:%s:%s", name, targetBase, token)
		handler, loaded := proxyCache.Load(cacheKey)
		if !loaded {
			h, err := proxy.NewWebSocketAwareHandlerWithHeaders(targetBase, stripPrefix, staticHeaders)
			if err != nil {
				log.Printf("Failed to create proxy for %s: %v", name, err)
				http.Error(w, "proxy error", http.StatusBadGateway)
				return
			}
			handler, _ = proxyCache.LoadOrStore(cacheKey, h)
		}

		handler.(http.Handler).ServeHTTP(w, r)
	})
}

func registerDisabledOpenClawRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/openclaw/status", writeOpenClawArray)
	mux.HandleFunc("/api/openclaw/teams", writeOpenClawArray)
	mux.HandleFunc("/api/openclaw/workers", writeOpenClawArray)
	mux.HandleFunc("/api/openclaw/rooms", writeOpenClawArray)
	mux.HandleFunc("/api/openclaw/rooms/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"OpenClaw feature disabled"}`, http.StatusServiceUnavailable)
	})
	mux.HandleFunc(
		"/embedded/openclaw/",
		serviceUnavailableHTMLHandler("OpenClaw", "OPENCLAW_ENABLED", "true"),
	)
}

func writeOpenClawArray(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`[]`))
}
