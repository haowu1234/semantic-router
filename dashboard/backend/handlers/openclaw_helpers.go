package handlers

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// rewriteLoopbackHost replaces 127.0.0.1 / localhost in a URL with the given
// container name so that inter-container traffic uses Docker DNS instead of
// loopback (which is unreachable across containers in bridge networks).
func rewriteLoopbackHost(rawURL, containerName string) string {
	if rawURL == "" || containerName == "" {
		return rawURL
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	host := u.Hostname()
	if host != "127.0.0.1" && host != "localhost" && host != "0.0.0.0" {
		return rawURL
	}
	port := u.Port()
	if port != "" {
		u.Host = containerName + ":" + port
	} else {
		u.Host = containerName
	}
	return u.String()
}

// --- Helpers ---

func sanitizeContainerName(raw string) string {
	cleaned := strings.ToLower(strings.TrimSpace(raw))
	cleaned = containerNameInvalidChars.ReplaceAllString(cleaned, "-")
	cleaned = strings.Trim(cleaned, "._-")
	if cleaned == "" {
		return ""
	}

	// Keep names bounded and still docker-friendly.
	const maxLen = 63
	if len(cleaned) > maxLen {
		cleaned = strings.Trim(cleaned[:maxLen], "._-")
	}
	if cleaned == "" {
		return ""
	}

	first := cleaned[0]
	if (first < 'a' || first > 'z') && (first < '0' || first > '9') {
		cleaned = "oc-" + cleaned
	}
	return cleaned
}

func sanitizeTeamID(raw string) string {
	return sanitizeContainerName(raw)
}

func sanitizeRoomID(raw string) string {
	return sanitizeContainerName(raw)
}

func normalizeRoleKind(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "leader":
		return "leader"
	default:
		return "worker"
	}
}

func deriveContainerName(requested, identityName string) string {
	if name := sanitizeContainerName(requested); name != "" {
		return name
	}
	if name := sanitizeContainerName(identityName); name != "" {
		return name
	}
	return "openclaw-vllm-sr"
}

func readIdentitySnapshot(dataDir string) identitySnapshot {
	wsDir := filepath.Join(dataDir, "workspace")
	snapshot := identitySnapshot{}

	identityContent, err := os.ReadFile(filepath.Join(wsDir, "IDENTITY.md"))
	if err == nil {
		for _, raw := range strings.Split(string(identityContent), "\n") {
			line := strings.TrimSpace(raw)
			switch {
			case strings.HasPrefix(line, "- **Name:**"):
				snapshot.Name = strings.TrimSpace(strings.TrimPrefix(line, "- **Name:**"))
			case strings.HasPrefix(line, "- **Creature:**"):
				snapshot.Role = strings.TrimSpace(strings.TrimPrefix(line, "- **Creature:**"))
			case strings.HasPrefix(line, "- **Vibe:**"):
				snapshot.Vibe = strings.TrimSpace(strings.TrimPrefix(line, "- **Vibe:**"))
			case strings.HasPrefix(line, "- **Emoji:**"):
				snapshot.Emoji = strings.TrimSpace(strings.TrimPrefix(line, "- **Emoji:**"))
			}
		}
	}

	soulContent, err := os.ReadFile(filepath.Join(wsDir, "SOUL.md"))
	if err == nil {
		lines := strings.Split(string(soulContent), "\n")
		capture := false
		var truths []string
		for _, raw := range lines {
			line := strings.TrimSpace(raw)
			if strings.HasPrefix(line, "## ") {
				if line == "## Core Truths" {
					capture = true
					continue
				}
				if capture {
					break
				}
			}
			if capture && line != "" {
				truths = append(truths, line)
			}
		}
		if len(truths) > 0 {
			snapshot.Principles = strings.Join(truths, " ")
		}
	}

	return snapshot
}

func writeJSONError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(map[string]string{"error": msg}); err != nil {
		log.Printf("openclaw: error encode error: %v", err)
	}
}

func generateToken(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "changeme-" + fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

func writeIdentityFiles(wsDir string, id IdentityConfig) error {
	var soulParts []string
	soulParts = append(soulParts, "# SOUL.md - Who You Are\n")
	if id.Name != "" || id.Role != "" {
		soulParts = append(soulParts, "## Core Identity\n")
		if id.Name != "" && id.Role != "" {
			soulParts = append(soulParts, fmt.Sprintf("You are **%s**, %s.\n", id.Name, id.Role))
		} else if id.Name != "" {
			soulParts = append(soulParts, fmt.Sprintf("You are **%s**.\n", id.Name))
		}
	}
	if id.Principles != "" {
		soulParts = append(soulParts, "## Core Truths\n")
		soulParts = append(soulParts, id.Principles+"\n")
	}
	if id.Boundaries != "" {
		soulParts = append(soulParts, "## Boundaries\n")
		soulParts = append(soulParts, id.Boundaries+"\n")
	}
	if id.Vibe != "" {
		soulParts = append(soulParts, "## Vibe\n")
		soulParts = append(soulParts, id.Vibe+"\n")
	}
	if err := os.WriteFile(filepath.Join(wsDir, "SOUL.md"), []byte(strings.Join(soulParts, "\n")), 0o644); err != nil {
		return err
	}

	var idParts []string
	idParts = append(idParts, "# IDENTITY.md - Who Am I?\n")
	if id.Name != "" {
		idParts = append(idParts, fmt.Sprintf("- **Name:** %s", id.Name))
	}
	if id.Role != "" {
		idParts = append(idParts, fmt.Sprintf("- **Creature:** %s", id.Role))
	}
	if id.Vibe != "" {
		idParts = append(idParts, fmt.Sprintf("- **Vibe:** %s", id.Vibe))
	}
	if id.Emoji != "" {
		idParts = append(idParts, fmt.Sprintf("- **Emoji:** %s", id.Emoji))
	}
	if err := os.WriteFile(filepath.Join(wsDir, "IDENTITY.md"), []byte(strings.Join(idParts, "\n")+"\n"), 0o644); err != nil {
		return err
	}

	var userParts []string
	userParts = append(userParts, "# USER.md - About Your Human\n")
	if id.UserName != "" {
		userParts = append(userParts, fmt.Sprintf("- **Name:** %s", id.UserName))
	}
	if id.UserNotes != "" {
		userParts = append(userParts, fmt.Sprintf("- **Notes:** %s", id.UserNotes))
	}
	return os.WriteFile(filepath.Join(wsDir, "USER.md"), []byte(strings.Join(userParts, "\n")+"\n"), 0o644)
}

func writeOpenClawConfig(path string, req ProvisionRequest) error {
	// Recover from stale state where a previous bad bind mount caused
	// openclaw.json to be created as a directory on host.
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		if removeErr := os.RemoveAll(path); removeErr != nil {
			return fmt.Errorf("failed to replace config directory %s with file: %w", path, removeErr)
		}
	} else if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to stat config path %s: %w", path, err)
	}

	cfg := map[string]interface{}{
		"models": map[string]interface{}{
			"providers": map[string]interface{}{
				"vllm": map[string]interface{}{
					"baseUrl": req.Container.ModelBaseURL,
					"apiKey":  req.Container.ModelAPIKey,
					"api":     "openai-completions",
					"headers": map[string]string{"x-authz-user-id": "openclaw-demo-user"},
					"models": []map[string]interface{}{
						{
							"id": req.Container.ModelName, "name": "SR Routed Model",
							"reasoning": false, "input": []string{"text", "image"},
							"cost":          map[string]interface{}{"input": 0.15, "output": 0.6, "cacheRead": 0, "cacheWrite": 0},
							"contextWindow": 30000, "maxTokens": 1024,
							"compat": map[string]string{"maxTokensField": "max_tokens"},
						},
					},
				},
			},
		},
		"agents": map[string]interface{}{
			"defaults": map[string]interface{}{
				"model":      map[string]string{"primary": "vllm/" + req.Container.ModelName},
				"workspace":  "/workspace",
				"compaction": map[string]string{"mode": "safeguard"},
			},
			"list": []map[string]interface{}{
				{"id": "vllm-sr", "default": true, "name": "vLLM-SR Powered Agent", "workspace": "/workspace"},
			},
		},
		"commands": map[string]interface{}{"native": "auto", "nativeSkills": "auto", "restart": true},
		"gateway": map[string]interface{}{
			"port": req.Container.GatewayPort,
			"auth": map[string]string{"mode": "token", "token": req.Container.AuthToken},
			"http": map[string]interface{}{
				"endpoints": map[string]interface{}{
					"chatCompletions": map[string]interface{}{"enabled": true},
					"responses":       map[string]interface{}{"enabled": true},
				},
			},
			"controlUi": map[string]interface{}{
				"dangerouslyDisableDeviceAuth": true,
				"allowInsecureAuth":            true,
				"allowedOrigins":               []string{"*"},
			},
		},
	}
	memoryBackend := strings.ToLower(strings.TrimSpace(req.Container.MemoryBackend))
	if memoryBackend == "" {
		memoryBackend = "local"
	}

	// OpenClaw v2 memory schema:
	// - memory.backend accepts "builtin" or "qmd"
	// - remote embedding config lives under agents.defaults.memorySearch
	switch memoryBackend {
	case "qmd":
		cfg["memory"] = map[string]interface{}{"backend": "qmd"}
	case "remote":
		cfg["memory"] = map[string]interface{}{"backend": "builtin"}

		memorySearch := map[string]interface{}{
			"enabled":  true,
			"provider": "openai",
		}

		remote := map[string]interface{}{}
		if baseURL := strings.TrimSpace(req.Container.MemoryBaseURL); baseURL != "" {
			remote["baseUrl"] = baseURL
		}
		if apiKey := strings.TrimSpace(req.Container.ModelAPIKey); apiKey != "" && apiKey != "not-needed" {
			remote["apiKey"] = apiKey
		}
		if len(remote) > 0 {
			memorySearch["remote"] = remote
		}

		agentsCfg, _ := cfg["agents"].(map[string]interface{})
		defaultsCfg, _ := agentsCfg["defaults"].(map[string]interface{})
		defaultsCfg["memorySearch"] = memorySearch
	default:
		// "local" (or unknown values) falls back to builtin memory without
		// remote embedding configuration.
		cfg["memory"] = map[string]interface{}{"backend": "builtin"}
	}
	if req.Container.BrowserEnabled {
		cfg["browser"] = map[string]interface{}{"enabled": true, "headless": true, "noSandbox": true}
	}

	// Matrix channel configuration for agent-to-agent and human-in-the-loop communication
	if req.Container.MatrixEnabled && req.Container.MatrixHomeserver != "" {
		domain := req.Container.MatrixDomain
		if domain == "" {
			domain = "matrix.vllm-sr.local"
		}
		adminUser := req.Container.MatrixAdminUser
		if adminUser == "" {
			adminUser = "admin"
		}

		// Determine if this is a leader agent (by role, not name)
		isLeader := req.RoleKind == "leader"

		// Derive the Matrix user ID for this agent
		matrixUsername := deriveMatrixUsername(req.Container.ContainerName, req.Identity.Name)
		matrixUserID := fmt.Sprintf("@%s:%s", matrixUsername, domain)

		// Build allowFrom list for DM policy
		// Leader can receive DMs from admin and system (Dashboard)
		// Worker cannot receive DMs (empty list, not null)
		dmAllowFrom := []string{}  // Initialize to empty slice to ensure JSON [] instead of null
		if isLeader {
			dmAllowFrom = []string{
				fmt.Sprintf("@%s:%s", adminUser, domain),
				fmt.Sprintf("@system:%s", domain),
			}
		}

		// 构建团队协作 SystemPrompt
		// 这个 prompt 教 AI 从 SOUL.md 获取团队身份信息
		teamSystemPrompt := `You are a collaborative AI agent in a team environment.

Your team identity and membership information is defined in SOUL.md under the <!-- TEAM:BEGIN --> section.
Read SOUL.md at session start to understand:
- Your team name and your role (leader/worker)
- Who the team leader is (use @leader or their @mention handle)
- All team members and their responsibilities

Coordination rules:
- Leaders can delegate tasks using @<worker-id> mentions
- Workers should report progress in plain text without @mentions
- Always coordinate effectively with your teammates`

		matrixChannel := map[string]interface{}{
			"enabled":    true,
			"homeserver": req.Container.MatrixHomeserver,
			"userId":     matrixUserID,
			"dm": map[string]interface{}{
				"policy":    "allowlist",
				"allowFrom": dmAllowFrom,
			},
			"groupPolicy": "allowlist",
			// groupAllowFrom: who can invite this agent to group rooms
			// Must include @system since Dashboard uses @system identity to invite agents
			"groupAllowFrom": []string{
				fmt.Sprintf("@%s:%s", adminUser, domain),
				fmt.Sprintf("@leader:%s", domain),
				fmt.Sprintf("@system:%s", domain),
			},
			"groups": map[string]interface{}{
				"*": map[string]interface{}{
					"allow":          true,
					"requireMention": true,
					"systemPrompt":   teamSystemPrompt,
				},
			},
		}

		// Add access token if provided
		if req.Container.MatrixAccessToken != "" {
			matrixChannel["accessToken"] = req.Container.MatrixAccessToken
		}

		cfg["channels"] = map[string]interface{}{
			"matrix": matrixChannel,
		}
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func generateDockerRunCmd(runtime string, req ProvisionRequest, dataDir string) string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	healthCmd := fmt.Sprintf(
		`node -e "fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"`,
		req.Container.GatewayPort,
	)

	// Build startup command.
	// With openclaw-matrix image, Matrix plugin is already built-in, no runtime installation needed.
	// The entrypoint doesn't start gateway automatically, so we can enable plugin first then start gateway.
	var startupCmd string
	if req.Container.MatrixEnabled {
		// Enable Matrix plugin first, then start gateway
		// Note: "plugins enable" only modifies config, doesn't require gateway to be running
		startupCmd = `openclaw plugins enable matrix; openclaw gateway --allow-unconfigured --bind lan`
	} else {
		startupCmd = `openclaw gateway --allow-unconfigured --bind lan`
	}

	// Mount the entire data directory as /config instead of a single file to avoid
	// EBUSY errors when OpenClaw uses atomic rename() to update the config.
	return fmt.Sprintf(`%s run -d \
  --name %s \
  --user 0:0 \
  --network %s \
  --health-cmd '%s' \
  --health-interval 30s \
  --health-timeout 5s \
  --health-start-period 15s \
  --health-retries 3 \
  -v %s/workspace:/workspace \
  -v %s:/config \
  -v %s:/state \
  -e OPENCLAW_CONFIG_PATH=/config/openclaw.json \
  -e OPENCLAW_STATE_DIR=/state \
  %s \
  %s`,
		runtime, req.Container.ContainerName, req.Container.NetworkMode, healthCmd,
		dataDir, dataDir, volumeName, req.Container.BaseImage, startupCmd)
}

func generateComposeYAML(req ProvisionRequest, dataDir string) string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	networkMode := req.Container.NetworkMode

	// Build command based on whether Matrix is enabled
	// Matrix plugin (@openclaw/matrix) is not bundled in base image; install at startup.
	// Use "plugins enable matrix" to explicitly activate the plugin before starting gateway.
	var commandYAML string
	if req.Container.MatrixEnabled {
		commandYAML = `command: ["sh", "-c", "set -e; if ! ls /state/plugins/@openclaw/matrix 2>/dev/null; then echo 'Installing @openclaw/matrix plugin...'; node openclaw.mjs plugins install @openclaw/matrix; fi; node openclaw.mjs plugins enable matrix; exec node openclaw.mjs gateway --allow-unconfigured --bind lan"]`
	} else {
		commandYAML = `command: ["node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan"]`
	}

	// For bridge network names (not "host" or "container:xxx"), use the networks syntax.
	// Mount the entire data directory as /config instead of a single file to avoid
	// EBUSY errors when OpenClaw uses atomic rename() to update the config.
	if networkMode != "" && networkMode != "host" && !strings.HasPrefix(networkMode, "container:") {
		return fmt.Sprintf(`services:
  openclaw:
    image: %s
    container_name: %s
    user: "0:0"
    networks:
      - %s
    volumes:
      - %s/workspace:/workspace
      - %s:/config
      - %s:/state
    environment:
      OPENCLAW_CONFIG_PATH: /config/openclaw.json
      OPENCLAW_STATE_DIR: /state
    healthcheck:
      test: ["CMD-SHELL", "node -e \"fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\""]
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3
    %s
    restart: unless-stopped

networks:
  %s:
    external: true

volumes:
  %s:
`, req.Container.BaseImage, req.Container.ContainerName, networkMode,
			dataDir, dataDir, volumeName,
			req.Container.GatewayPort,
			commandYAML,
			networkMode, volumeName)
	}

	// Host or container:xxx network mode - also use directory mount for config
	return fmt.Sprintf(`services:
  openclaw:
    image: %s
    container_name: %s
    user: "0:0"
    network_mode: %s
    volumes:
      - %s/workspace:/workspace
      - %s:/config
      - %s:/state
    environment:
      OPENCLAW_CONFIG_PATH: /config/openclaw.json
      OPENCLAW_STATE_DIR: /state
    healthcheck:
      test: ["CMD-SHELL", "node -e \"fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\""]
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3
    %s
    restart: unless-stopped

volumes:
  %s:
`, req.Container.BaseImage, req.Container.ContainerName, networkMode,
		dataDir, dataDir, volumeName,
		req.Container.GatewayPort,
		commandYAML,
		volumeName)
}

func agentsMdContent() string {
	return `# AGENTS.md - Your Workspace

This folder is home. Treat it that way.

## Every Session

Before doing anything else:

1. Read ` + "`SOUL.md`" + ` — this is who you are
2. Read ` + "`IDENTITY.md`" + ` — your profile, vibe, and persona details
3. Read ` + "`USER.md`" + ` — this is who you're helping
4. Read ` + "`memory/`" + ` for recent context

## Memory

You wake up fresh each session. These files are your continuity:

- **Daily notes:** ` + "`memory/YYYY-MM-DD.md`" + ` — raw logs of what happened
- **Skills:** ` + "`skills/*/SKILL.md`" + ` — your specialized abilities

## Safety

- Don't exfiltrate private data
- Don't run destructive commands without asking
- When in doubt, ask

## Tools

Skills provide your tools. When you need one, check its SKILL.md.
`
}

// deriveMatrixUsername derives a Matrix username from container name and identity name.
// The username is sanitized to be valid for Matrix user IDs.
func deriveMatrixUsername(containerName, identityName string) string {
	// Prefer identity name if available, otherwise use container name
	raw := strings.TrimSpace(identityName)
	if raw == "" {
		raw = containerName
	}

	// Sanitize for Matrix username (lowercase, alphanumeric, dots, dashes, underscores)
	// Matrix user localpart: a-z, 0-9, ., _, =, -, /
	cleaned := strings.ToLower(raw)
	var result strings.Builder
	for _, r := range cleaned {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '.' || r == '_' || r == '-' {
			result.WriteRune(r)
		} else if r == ' ' || r == ':' {
			result.WriteRune('_')
		}
	}

	username := result.String()
	if username == "" {
		username = "worker"
	}

	// Ensure username doesn't start with underscore or dot
	username = strings.TrimLeft(username, "._")
	if username == "" {
		username = "worker"
	}

	// Limit length (Matrix allows up to 255, but keep it reasonable)
	const maxLen = 32
	if len(username) > maxLen {
		username = username[:maxLen]
	}

	return username
}

// ensureMatrixUserAndGetToken registers a Matrix user (if not exists) and returns an access token.
// This uses the shared registration token for user creation.
func (h *OpenClawHandler) ensureMatrixUserAndGetToken(username string) (string, error) {
	if h.matrixClient == nil {
		return "", fmt.Errorf("matrix client not initialized")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Generate a deterministic password for the agent based on username and a secret
	// In production, consider using a more secure password derivation
	password := generateAgentPassword(username)

	// Try to login first (user may already exist)
	token, err := h.matrixClient.LoginUser(ctx, username, password)
	if err == nil {
		log.Printf("Matrix user %q already exists, logged in successfully", username)
		return token, nil
	}

	// Login failed, try to register
	log.Printf("Matrix login failed for %q, attempting registration: %v", username, err)

	// RegisterUser now returns access token directly
	token, regErr := h.matrixClient.RegisterUser(ctx, username, password)
	if regErr != nil {
		// Check if user already exists (M_USER_IN_USE)
		if strings.Contains(regErr.Error(), "M_USER_IN_USE") {
			// User exists but password might be different, this is an error state
			return "", fmt.Errorf("matrix user %q exists but login failed (password mismatch?): %w", username, err)
		}
		return "", fmt.Errorf("failed to register matrix user %q: %w", username, regErr)
	}

	log.Printf("Matrix user %q registered successfully", username)
	return token, nil
}

// generateAgentPassword generates a deterministic password for an agent.
// Uses the registration token as a seed to create reproducible passwords.
func generateAgentPassword(username string) string {
	regToken := strings.TrimSpace(os.Getenv("MATRIX_REG_TOKEN"))
	if regToken == "" {
		regToken = "default-agent-password-seed"
	}

	// Simple deterministic password: hash of username + regToken
	// This ensures the same agent always gets the same password
	combined := username + ":" + regToken

	// Use a simple hash (in production, use HMAC or similar)
	h := 0
	for _, c := range combined {
		h = 31*h + int(c)
	}

	// Generate password string
	return fmt.Sprintf("agent-%s-%x", username, uint32(h))
}
