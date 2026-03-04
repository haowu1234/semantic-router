package handlers

import "net/http"

// ReadonlyChecker provides unified readonly + invite checking
type ReadonlyChecker struct {
	ReadonlyMode bool
	InviteSecret string
}

// NewReadonlyChecker creates a new checker
func NewReadonlyChecker(readonlyMode bool, inviteSecret string) *ReadonlyChecker {
	return &ReadonlyChecker{
		ReadonlyMode: readonlyMode,
		InviteSecret: inviteSecret,
	}
}

// IsBlocked returns true if the request should be blocked
// Returns false if:
//   - readonly mode is disabled, OR
//   - readonly mode is enabled but request has valid invite code
func (c *ReadonlyChecker) IsBlocked(r *http.Request) bool {
	if !c.ReadonlyMode {
		return false
	}
	// In readonly mode, check for valid invite
	if c.InviteSecret != "" && HasValidInvite(r, c.InviteSecret) {
		return false
	}
	return true
}

// WriteBlockedError writes a standardized blocked response
func (c *ReadonlyChecker) WriteBlockedError(w http.ResponseWriter) {
	msg := "Operation not allowed in readonly mode"
	if c.InviteSecret != "" {
		msg += " – use an invite code to unlock editing"
	}
	http.Error(w, msg, http.StatusForbidden)
}

// CheckAndBlock is a convenience method that checks and writes error if blocked
// Returns true if blocked (caller should return), false if allowed to proceed
func (c *ReadonlyChecker) CheckAndBlock(w http.ResponseWriter, r *http.Request) bool {
	if c.IsBlocked(r) {
		c.WriteBlockedError(w)
		return true
	}
	return false
}

// IsReadonlyBlocked is a standalone function for handlers that don't use ReadonlyChecker
// Returns true if the request should be blocked due to readonly mode
func IsReadonlyBlocked(readonlyMode bool, inviteSecret string, r *http.Request) bool {
	if !readonlyMode {
		return false
	}
	if inviteSecret != "" && HasValidInvite(r, inviteSecret) {
		return false
	}
	return true
}

// WriteReadonlyError writes a standardized readonly error response
func WriteReadonlyError(w http.ResponseWriter, inviteSecret string) {
	msg := "Operation not allowed in readonly mode"
	if inviteSecret != "" {
		msg += " – use an invite code to unlock editing"
	}
	http.Error(w, msg, http.StatusForbidden)
}
