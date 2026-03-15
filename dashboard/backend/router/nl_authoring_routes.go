package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/nlauthor"
)

func registerNLAuthoringRoutes(mux *http.ServeMux, cfg *config.Config) {
	manifest := nlauthor.DefaultSchemaManifest()
	service := nlauthor.NewPreviewService(manifest)

	mux.HandleFunc("/api/builder/nl/capabilities", handlers.NLAuthoringCapabilitiesHandler(cfg, service))
	mux.HandleFunc("/api/builder/nl/schema", handlers.NLAuthoringSchemaHandler(service))
	mux.HandleFunc("/api/builder/nl/sessions", handlers.NLAuthoringSessionsHandler(cfg, service))
	mux.HandleFunc("/api/builder/nl/sessions/", handlers.NLAuthoringSessionTurnsHandler(cfg, service))
	log.Printf("NL authoring API endpoints registered: /api/builder/nl/capabilities, /api/builder/nl/schema, /api/builder/nl/sessions, /api/builder/nl/sessions/{id}/turns")
}
