package nlauthor

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

const defaultSessionTTL = 30 * time.Minute

type ErrorCode string

const (
	ErrorCodeInvalidArgument ErrorCode = "invalid_argument"
	ErrorCodeForbidden       ErrorCode = "forbidden"
	ErrorCodeNotFound        ErrorCode = "not_found"
	ErrorCodeInternal        ErrorCode = "internal"
)

// ServiceError is the domain error returned by the NL authoring service.
type ServiceError struct {
	Code    ErrorCode
	Message string
}

func (e *ServiceError) Error() string {
	return e.Message
}

// Service owns session lifecycle, schema version checks, and planner dispatch.
type Service struct {
	manifest   SchemaManifest
	store      SessionStore
	planner    Planner
	sessionTTL time.Duration
	now        func() time.Time
}

func NewService(manifest SchemaManifest, store SessionStore, planner Planner, sessionTTL time.Duration) *Service {
	if manifest.Version == "" {
		manifest = DefaultSchemaManifest()
	}
	if store == nil {
		store = NewInMemorySessionStore()
	}
	if planner == nil {
		planner = UnavailablePlanner{}
	}
	if sessionTTL <= 0 {
		sessionTTL = defaultSessionTTL
	}

	return &Service{
		manifest:   manifest,
		store:      store,
		planner:    planner,
		sessionTTL: sessionTTL,
		now: func() time.Time {
			return time.Now().UTC()
		},
	}
}

func NewPreviewService(manifest SchemaManifest) *Service {
	planner := NewPreviewPlanner(manifest)
	return NewService(manifest, NewInMemorySessionStore(), planner, defaultSessionTTL)
}

func (s *Service) Capabilities(readonlyMode bool) Capabilities {
	capabilities := DefaultCapabilities(readonlyMode)
	support := s.planner.Support()
	capabilities.SchemaVersion = s.manifest.Version
	capabilities.PlannerAvailable = s.planner.Available()
	capabilities.PlannerBackend = s.planner.BackendName()
	capabilities.SupportsClarification = s.planner.SupportsClarification()
	capabilities.SupportedOperations = append([]OperationMode{}, support.Operations...)
	capabilities.SupportedConstructs = append([]ConstructKind{}, support.Constructs...)
	capabilities.SupportedSignalTypes = append([]string{}, support.SignalTypes...)
	capabilities.SupportedPluginTypes = append([]string{}, support.PluginTypes...)
	capabilities.SupportedBackendTypes = append([]string{}, support.BackendTypes...)
	capabilities.SupportedAlgoTypes = append([]string{}, support.AlgorithmTypes...)
	return capabilities
}

func (s *Service) Schema() SchemaManifest {
	return s.manifest
}

func (s *Service) CreateSession(request SessionCreateRequest, readonlyMode bool, ownerUserID string) (SessionCreateResponse, error) {
	schemaVersion, err := s.resolveSchemaVersion(request.SchemaVersion)
	if err != nil {
		return SessionCreateResponse{}, err
	}

	now := s.now()
	session := Session{
		ID:            uuid.NewString(),
		SchemaVersion: schemaVersion,
		OwnerUserID:   strings.TrimSpace(ownerUserID),
		Context:       request.Context,
		CreatedAt:     now,
		UpdatedAt:     now,
		ExpiresAt:     now.Add(s.sessionTTL),
	}
	if err := s.store.Create(session); err != nil {
		return SessionCreateResponse{}, &ServiceError{
			Code:    ErrorCodeInternal,
			Message: "failed to create NL authoring session",
		}
	}

	return SessionCreateResponse{
		SessionID:     session.ID,
		SchemaVersion: schemaVersion,
		ExpiresAt:     session.ExpiresAt.Format(time.RFC3339),
		Capabilities:  s.Capabilities(readonlyMode),
	}, nil
}

func (s *Service) RunTurn(ctx context.Context, sessionID string, request TurnRequest, _ bool, ownerUserID string) (TurnResponse, error) {
	trimmedSessionID := strings.TrimSpace(sessionID)
	if trimmedSessionID == "" {
		return TurnResponse{}, &ServiceError{
			Code:    ErrorCodeInvalidArgument,
			Message: "sessionId is required",
		}
	}
	prompt := strings.TrimSpace(request.Prompt)
	if prompt == "" {
		return TurnResponse{}, &ServiceError{
			Code:    ErrorCodeInvalidArgument,
			Message: "prompt is required",
		}
	}

	now := s.now()
	session, err := s.store.Get(trimmedSessionID, now)
	if err != nil {
		if errors.Is(err, errSessionNotFound) {
			return TurnResponse{}, &ServiceError{
				Code:    ErrorCodeNotFound,
				Message: "session not found or expired",
			}
		}
		return TurnResponse{}, &ServiceError{
			Code:    ErrorCodeInternal,
			Message: "failed to load NL authoring session",
		}
	}
	if session.OwnerUserID != "" && strings.TrimSpace(ownerUserID) != session.OwnerUserID {
		return TurnResponse{}, &ServiceError{
			Code:    ErrorCodeForbidden,
			Message: "session does not belong to the current user",
		}
	}

	schemaVersion, err := s.resolveSchemaVersion(request.SchemaVersion)
	if err != nil {
		return TurnResponse{}, err
	}
	if schemaVersion != session.SchemaVersion {
		return TurnResponse{}, &ServiceError{
			Code:    ErrorCodeInvalidArgument,
			Message: fmt.Sprintf("session schema version %q does not match request schema version %q", session.SchemaVersion, schemaVersion),
		}
	}

	request.Prompt = prompt
	request.SchemaVersion = schemaVersion
	if sessionContextProvided(request.Context) {
		session.Context = request.Context
	}
	session.UpdatedAt = now
	session.ExpiresAt = now.Add(s.sessionTTL)
	if updateErr := s.store.Update(session, now); updateErr != nil {
		if errors.Is(updateErr, errSessionNotFound) {
			return TurnResponse{}, &ServiceError{
				Code:    ErrorCodeNotFound,
				Message: "session not found or expired",
			}
		}
		return TurnResponse{}, &ServiceError{
			Code:    ErrorCodeInternal,
			Message: "failed to persist NL authoring session",
		}
	}

	result, err := s.planner.Plan(ctx, session, request)
	if err != nil {
		var serviceErr *ServiceError
		if errors.As(err, &serviceErr) {
			return TurnResponse{}, serviceErr
		}
		return TurnResponse{}, &ServiceError{
			Code:    ErrorCodeInternal,
			Message: "planner execution failed",
		}
	}
	result = normalizePlannerResult(result, s.manifest)

	return TurnResponse{
		SessionID:     session.ID,
		TurnID:        uuid.NewString(),
		SchemaVersion: session.SchemaVersion,
		ExpiresAt:     session.ExpiresAt.Format(time.RFC3339),
		Result:        result,
	}, nil
}

func (s *Service) resolveSchemaVersion(requested string) (string, error) {
	schemaVersion := strings.TrimSpace(requested)
	if schemaVersion == "" {
		schemaVersion = s.manifest.Version
	}
	if schemaVersion == "" {
		schemaVersion = SchemaVersion
	}
	if schemaVersion != s.manifest.Version {
		return "", &ServiceError{
			Code:    ErrorCodeInvalidArgument,
			Message: fmt.Sprintf("unsupported schema version %q, expected %q", schemaVersion, s.manifest.Version),
		}
	}
	return schemaVersion, nil
}

func sessionContextProvided(context SessionContext) bool {
	return strings.TrimSpace(context.BaseDSL) != "" || context.Symbols != nil || len(context.Diagnostics) > 0
}
