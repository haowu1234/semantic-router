package handlers

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestVerifyInviteCode(t *testing.T) {
	secret := "test-secret-key-32bytes-long!!"

	tests := []struct {
		name       string
		code       string
		secret     string
		wantValid  bool
		wantScope  string
		wantNote   string
	}{
		{
			name:      "empty secret",
			code:      "invite-xxx.yyy",
			secret:    "",
			wantValid: false,
		},
		{
			name:      "invalid format - no dot",
			code:      "invite-xxxyyy",
			secret:    secret,
			wantValid: false,
		},
		{
			name:      "invalid format - empty parts",
			code:      "invite-.",
			secret:    secret,
			wantValid: false,
		},
		{
			name:      "invalid signature",
			code:      "invite-eyJleHAiOjAsInNjb3BlIjoid3JpdGUifQ.invalidsig",
			secret:    secret,
			wantValid: false,
		},
		{
			name:      "invalid base64 payload",
			code:      "invite-!!!invalid!!!.abc123",
			secret:    secret,
			wantValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload, valid := VerifyInviteCode(tt.code, tt.secret)
			if valid != tt.wantValid {
				t.Errorf("VerifyInviteCode() valid = %v, want %v", valid, tt.wantValid)
			}
			if tt.wantValid {
				if payload.Scope != tt.wantScope {
					t.Errorf("VerifyInviteCode() scope = %v, want %v", payload.Scope, tt.wantScope)
				}
				if payload.Note != tt.wantNote {
					t.Errorf("VerifyInviteCode() note = %v, want %v", payload.Note, tt.wantNote)
				}
			}
		})
	}
}

func TestVerifyInviteCode_ValidCode(t *testing.T) {
	secret := "test-secret-key-32bytes-long!!"

	// Create a valid payload that never expires
	payload := InvitePayload{
		Exp:   0, // Never expires
		Scope: "write",
		Note:  "test-user",
	}

	// Generate valid code using same logic as CLI
	code := generateTestInviteCode(payload, secret)

	// Test verification
	result, valid := VerifyInviteCode(code, secret)
	if !valid {
		t.Fatalf("VerifyInviteCode() should return valid=true for valid code")
	}
	if result.Scope != "write" {
		t.Errorf("Scope = %v, want 'write'", result.Scope)
	}
	if result.Note != "test-user" {
		t.Errorf("Note = %v, want 'test-user'", result.Note)
	}
}

func TestVerifyInviteCode_ExpiredCode(t *testing.T) {
	secret := "test-secret-key-32bytes-long!!"

	// Create an expired payload
	payload := InvitePayload{
		Exp:   time.Now().Unix() - 3600, // Expired 1 hour ago
		Scope: "write",
	}

	code := generateTestInviteCode(payload, secret)

	// Should be invalid due to expiration
	_, valid := VerifyInviteCode(code, secret)
	if valid {
		t.Errorf("VerifyInviteCode() should return valid=false for expired code")
	}
}

func TestVerifyInviteCode_FutureExpiration(t *testing.T) {
	secret := "test-secret-key-32bytes-long!!"

	// Create a payload that expires in the future
	payload := InvitePayload{
		Exp:   time.Now().Unix() + 3600, // Expires in 1 hour
		Scope: "admin",
		Note:  "admin-user",
	}

	code := generateTestInviteCode(payload, secret)

	result, valid := VerifyInviteCode(code, secret)
	if !valid {
		t.Fatalf("VerifyInviteCode() should return valid=true for non-expired code")
	}
	if result.Scope != "admin" {
		t.Errorf("Scope = %v, want 'admin'", result.Scope)
	}
}

func TestVerifyInviteCode_WrongSecret(t *testing.T) {
	secret := "test-secret-key-32bytes-long!!"
	wrongSecret := "wrong-secret-key-different!!!!!"

	payload := InvitePayload{
		Exp:   0,
		Scope: "write",
	}

	code := generateTestInviteCode(payload, secret)

	// Should fail with wrong secret
	_, valid := VerifyInviteCode(code, wrongSecret)
	if valid {
		t.Errorf("VerifyInviteCode() should return valid=false for wrong secret")
	}
}

func TestHasValidInvite(t *testing.T) {
	secret := "test-secret-key-32bytes-long!!"

	payload := InvitePayload{
		Exp:   0,
		Scope: "write",
	}
	validCode := generateTestInviteCode(payload, secret)

	tests := []struct {
		name      string
		cookie    *http.Cookie
		secret    string
		wantValid bool
	}{
		{
			name:      "no cookie",
			cookie:    nil,
			secret:    secret,
			wantValid: false,
		},
		{
			name:      "empty secret",
			cookie:    &http.Cookie{Name: "sr-invite", Value: validCode},
			secret:    "",
			wantValid: false,
		},
		{
			name:      "valid cookie",
			cookie:    &http.Cookie{Name: "sr-invite", Value: validCode},
			secret:    secret,
			wantValid: true,
		},
		{
			name:      "invalid cookie value",
			cookie:    &http.Cookie{Name: "sr-invite", Value: "invalid-code"},
			secret:    secret,
			wantValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			if tt.cookie != nil {
				req.AddCookie(tt.cookie)
			}

			valid := HasValidInvite(req, tt.secret)
			if valid != tt.wantValid {
				t.Errorf("HasValidInvite() = %v, want %v", valid, tt.wantValid)
			}
		})
	}
}

func TestVerifyInviteHandler(t *testing.T) {
	secret := "test-secret-key-32bytes-long!!"

	payload := InvitePayload{
		Exp:   0,
		Scope: "write",
		Note:  "test",
	}
	validCode := generateTestInviteCode(payload, secret)

	tests := []struct {
		name       string
		method     string
		body       string
		secret     string
		wantStatus int
		wantCookie bool
	}{
		{
			name:       "wrong method",
			method:     http.MethodGet,
			body:       `{"code":"test"}`,
			secret:     secret,
			wantStatus: http.StatusMethodNotAllowed,
			wantCookie: false,
		},
		{
			name:       "empty secret",
			method:     http.MethodPost,
			body:       `{"code":"test"}`,
			secret:     "",
			wantStatus: http.StatusNotImplemented,
			wantCookie: false,
		},
		{
			name:       "invalid body",
			method:     http.MethodPost,
			body:       `{invalid json`,
			secret:     secret,
			wantStatus: http.StatusBadRequest,
			wantCookie: false,
		},
		{
			name:       "invalid code",
			method:     http.MethodPost,
			body:       `{"code":"invalid-code"}`,
			secret:     secret,
			wantStatus: http.StatusForbidden,
			wantCookie: false,
		},
		{
			name:       "valid code",
			method:     http.MethodPost,
			body:       `{"code":"` + validCode + `"}`,
			secret:     secret,
			wantStatus: http.StatusOK,
			wantCookie: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := VerifyInviteHandler(tt.secret)

			req := httptest.NewRequest(tt.method, "/api/invite/verify", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if w.Code != tt.wantStatus {
				t.Errorf("Status = %v, want %v", w.Code, tt.wantStatus)
			}

			// Check cookie
			cookies := w.Result().Cookies()
			hasCookie := false
			for _, c := range cookies {
				if c.Name == "sr-invite" && c.Value != "" {
					hasCookie = true
					break
				}
			}
			if hasCookie != tt.wantCookie {
				t.Errorf("HasCookie = %v, want %v", hasCookie, tt.wantCookie)
			}
		})
	}
}

func TestLogoutInviteHandler(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		wantStatus int
	}{
		{
			name:       "wrong method",
			method:     http.MethodGet,
			wantStatus: http.StatusMethodNotAllowed,
		},
		{
			name:       "valid logout",
			method:     http.MethodPost,
			wantStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := LogoutInviteHandler()

			req := httptest.NewRequest(tt.method, "/api/invite/logout", nil)
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if w.Code != tt.wantStatus {
				t.Errorf("Status = %v, want %v", w.Code, tt.wantStatus)
			}

			if tt.wantStatus == http.StatusOK {
				// Check that cookie is cleared
				cookies := w.Result().Cookies()
				for _, c := range cookies {
					if c.Name == "sr-invite" {
						if c.MaxAge != -1 {
							t.Errorf("Cookie MaxAge = %v, want -1 (delete)", c.MaxAge)
						}
					}
				}
			}
		})
	}
}

// generateTestInviteCode generates a valid invite code for testing
// This mirrors the generation logic in the CLI
func generateTestInviteCode(payload InvitePayload, secret string) string {
	payloadJSON, _ := json.Marshal(payload)
	payloadB64 := base64.RawURLEncoding.EncodeToString(payloadJSON)

	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(payloadB64))
	sig := hex.EncodeToString(mac.Sum(nil))

	return "invite-" + payloadB64 + "." + sig
}

// TestCrossLanguageCompatibility tests that codes generated by Python CLI
// can be verified by Go backend (and vice versa)
func TestCrossLanguageCompatibility(t *testing.T) {
	// This code was generated by Python CLI:
	// python3 -c "from cli.commands.invite import generate_invite_code; print(generate_invite_code('cross-lang-test-secret-key-32!', 0, 'write', 'cross-lang-test'))"
	pythonGeneratedCode := "invite-eyJleHAiOjAsInNjb3BlIjoid3JpdGUiLCJub3RlIjoiY3Jvc3MtbGFuZy10ZXN0In0.ab10a6c6eb6354331c9ac263ed394710fb3268a18cc98b8a730d58cad3078604"
	secret := "cross-lang-test-secret-key-32!"

	payload, valid := VerifyInviteCode(pythonGeneratedCode, secret)
	if !valid {
		t.Fatalf("Go backend should verify Python-generated code")
	}
	if payload.Scope != "write" {
		t.Errorf("Scope = %v, want 'write'", payload.Scope)
	}
	if payload.Note != "cross-lang-test" {
		t.Errorf("Note = %v, want 'cross-lang-test'", payload.Note)
	}
	if payload.Exp != 0 {
		t.Errorf("Exp = %v, want 0 (never expires)", payload.Exp)
	}
}
