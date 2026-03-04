package handlers

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

// InvitePayload represents the decoded invite code payload
type InvitePayload struct {
	Exp   int64  `json:"exp"`            // Expiration time (Unix timestamp), 0 = never expires
	Scope string `json:"scope"`          // Permission scope: "write" | "admin"
	Note  string `json:"note,omitempty"` // Optional note for auditing
}

// VerifyInviteCode validates the invite code signature and expiration
// Returns the payload and validity status
func VerifyInviteCode(code, secret string) (*InvitePayload, bool) {
	if secret == "" {
		return nil, false
	}

	// Format: invite-{base64_payload}.{hex_signature}
	code = strings.TrimPrefix(code, "invite-")
	parts := strings.SplitN(code, ".", 2)
	if len(parts) != 2 {
		return nil, false
	}
	payloadB64, sigHex := parts[0], parts[1]

	// Verify HMAC signature (constant-time comparison)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(payloadB64))
	expectedSig := hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(sigHex), []byte(expectedSig)) {
		return nil, false
	}

	// Decode payload (URL-safe base64 without padding)
	payloadBytes, err := base64.RawURLEncoding.DecodeString(payloadB64)
	if err != nil {
		return nil, false
	}
	var payload InvitePayload
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return nil, false
	}

	// Check expiration (exp=0 means never expires)
	if payload.Exp > 0 && time.Now().Unix() > payload.Exp {
		return nil, false
	}

	return &payload, true
}

// HasValidInvite checks if the request has a valid invite code in cookie
func HasValidInvite(r *http.Request, secret string) bool {
	if secret == "" {
		return false
	}
	cookie, err := r.Cookie("sr-invite")
	if err != nil {
		return false
	}
	_, valid := VerifyInviteCode(cookie.Value, secret)
	return valid
}

// VerifyInviteHandler handles POST /api/invite/verify
func VerifyInviteHandler(secret string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Handle CORS
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			writeInviteJSONError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if secret == "" {
			writeInviteJSONError(w, "invite feature not enabled", http.StatusNotImplemented)
			return
		}

		var req struct {
			Code string `json:"code"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeInviteJSONError(w, "invalid request body", http.StatusBadRequest)
			return
		}

		payload, valid := VerifyInviteCode(req.Code, secret)
		if !valid {
			writeInviteJSONError(w, "invalid or expired invite code", http.StatusForbidden)
			return
		}

		// Calculate cookie MaxAge
		maxAge := 7 * 24 * 3600 // Default 7 days
		if payload.Exp > 0 {
			remaining := payload.Exp - time.Now().Unix()
			if remaining > 0 {
				maxAge = int(remaining)
			} else {
				// Already expired (shouldn't reach here due to earlier check)
				writeInviteJSONError(w, "invite code has expired", http.StatusForbidden)
				return
			}
		}

		// Set cookie with full invite code (preserve format for verification)
		http.SetCookie(w, &http.Cookie{
			Name:     "sr-invite",
			Value:    req.Code, // Keep full format including "invite-" prefix
			Path:     "/",
			MaxAge:   maxAge,
			HttpOnly: true,
			SameSite: http.SameSiteStrictMode,
			Secure:   r.TLS != nil, // Set Secure flag if HTTPS
		})

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"ok":    true,
			"scope": payload.Scope,
			"exp":   payload.Exp,
			"note":  payload.Note,
		})
	}
}

// LogoutInviteHandler handles POST /api/invite/logout (clear invite cookie)
func LogoutInviteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			writeInviteJSONError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Clear the cookie
		http.SetCookie(w, &http.Cookie{
			Name:     "sr-invite",
			Value:    "",
			Path:     "/",
			MaxAge:   -1, // Delete cookie
			HttpOnly: true,
			SameSite: http.SameSiteStrictMode,
		})

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]bool{"ok": true})
	}
}

// writeInviteJSONError writes a JSON error response
func writeInviteJSONError(w http.ResponseWriter, message string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}
