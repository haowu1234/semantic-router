package handlers

import (
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/nlauthor"
)

const nlAuthoringSessionsPrefix = "/api/builder/nl/sessions/"

type nlAuthoringErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// NLAuthoringSessionsHandler creates Builder-scoped NL authoring sessions.
func NLAuthoringSessionsHandler(cfg *config.Config, service *nlauthor.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var request nlauthor.SessionCreateRequest
		if err := decodeJSONStrict(r, &request); err != nil {
			if errors.Is(err, io.EOF) {
				request = nlauthor.SessionCreateRequest{}
			} else {
				writeJSONStatus(w, http.StatusBadRequest, nlAuthoringErrorResponse{
					Error:   string(nlauthor.ErrorCodeInvalidArgument),
					Message: "invalid request body: " + err.Error(),
				})
				return
			}
		}

		response, err := service.CreateSession(request, effectiveReadonlyMode(cfg, r), requestOwnerUserID(r))
		if err != nil {
			writeNLAuthoringServiceError(w, err)
			return
		}

		writeJSONStatus(w, http.StatusCreated, response)
	}
}

// NLAuthoringSessionTurnsHandler runs one planner turn for a Builder NL session.
func NLAuthoringSessionTurnsHandler(cfg *config.Config, service *nlauthor.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		sessionID, ok := nlAuthoringSessionIDFromTurnsPath(r.URL.Path)
		if !ok {
			http.NotFound(w, r)
			return
		}

		var request nlauthor.TurnRequest
		if err := decodeJSONStrict(r, &request); err != nil {
			writeJSONStatus(w, http.StatusBadRequest, nlAuthoringErrorResponse{
				Error:   string(nlauthor.ErrorCodeInvalidArgument),
				Message: "invalid request body: " + err.Error(),
			})
			return
		}

		response, err := service.RunTurn(r.Context(), sessionID, request, effectiveReadonlyMode(cfg, r), requestOwnerUserID(r))
		if err != nil {
			writeNLAuthoringServiceError(w, err)
			return
		}

		writeJSONStatus(w, http.StatusOK, response)
	}
}

func nlAuthoringSessionIDFromTurnsPath(path string) (string, bool) {
	if !strings.HasPrefix(path, nlAuthoringSessionsPrefix) || !strings.HasSuffix(path, "/turns") {
		return "", false
	}

	sessionID := strings.TrimSuffix(strings.TrimPrefix(path, nlAuthoringSessionsPrefix), "/turns")
	sessionID = strings.Trim(sessionID, "/")
	if sessionID == "" || strings.Contains(sessionID, "/") {
		return "", false
	}
	return sessionID, true
}

func writeNLAuthoringServiceError(w http.ResponseWriter, err error) {
	statusCode := http.StatusInternalServerError
	errorCode := string(nlauthor.ErrorCodeInternal)
	message := "internal server error"

	var serviceErr *nlauthor.ServiceError
	if errors.As(err, &serviceErr) {
		errorCode = string(serviceErr.Code)
		message = serviceErr.Message
		switch serviceErr.Code {
		case nlauthor.ErrorCodeInvalidArgument:
			statusCode = http.StatusBadRequest
		case nlauthor.ErrorCodeForbidden:
			statusCode = http.StatusForbidden
		case nlauthor.ErrorCodeNotFound:
			statusCode = http.StatusNotFound
		default:
			statusCode = http.StatusInternalServerError
		}
	}

	writeJSONStatus(w, statusCode, nlAuthoringErrorResponse{
		Error:   errorCode,
		Message: message,
	})
}
