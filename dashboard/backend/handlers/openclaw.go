package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

var containerNameInvalidChars = regexp.MustCompile(`[^a-z0-9_.-]+`)

// --- Registry ---

type ContainerEntry struct {
	Name            string `json:"name"`
	Port            int    `json:"port"`
	Image           string `json:"image"`
	Token           string `json:"token"`
	DataDir         string `json:"dataDir"`
	CreatedAt       string `json:"createdAt"`
	TeamID          string `json:"teamId,omitempty"`
	TeamName        string `json:"teamName,omitempty"`
	AgentName       string `json:"agentName,omitempty"`
	AgentEmoji      string `json:"agentEmoji,omitempty"`
	AgentRole       string `json:"agentRole,omitempty"`
	AgentVibe       string `json:"agentVibe,omitempty"`
	AgentPrinciples string `json:"agentPrinciples,omitempty"`
	RoleKind        string `json:"roleKind,omitempty"`
}

type TeamEntry struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	Vibe          string `json:"vibe,omitempty"`
	Role          string `json:"role,omitempty"`
	Principal     string `json:"principal,omitempty"`
	Description   string `json:"description,omitempty"`
	LeaderID      string `json:"leaderId,omitempty"`
	MatrixRoomID  string `json:"matrixRoomId,omitempty"` // Actual Matrix room ID (e.g., !abc123:domain)
	CreatedAt     string `json:"createdAt"`
	UpdatedAt     string `json:"updatedAt"`
}

type OpenClawHandler struct {
	dataDir          string
	readOnly         bool
	routerConfigPath string
	mu               sync.RWMutex
	roomSSEClients   sync.Map
	roomSSELastEvent sync.Map
	roomAutomationMu sync.Map
	// Matrix client for registering worker agents (optional, nil if Matrix disabled)
	matrixClient *MatrixClient
	matrixDomain string
	// Matrix bridge for hybrid native/matrix communication (optional, nil if Matrix disabled)
	matrixBridge *MatrixBridge
}

func NewOpenClawHandler(dataDir string, readOnly bool) *OpenClawHandler {
	return &OpenClawHandler{dataDir: dataDir, readOnly: readOnly}
}

// SetMatrixClient sets the Matrix client for dynamic worker user registration.
// This should be called when MATRIX_ENABLED=true during initialization.
func (h *OpenClawHandler) SetMatrixClient(client *MatrixClient, domain string) {
	h.matrixClient = client
	h.matrixDomain = domain
}

// SetMatrixBridge sets the Matrix bridge for Matrix-only communication.
// NOTE: Native store syncing has been removed - all communication goes through Matrix
func (h *OpenClawHandler) SetMatrixBridge(bridge *MatrixBridge) {
	h.matrixBridge = bridge
	// 初始化已有 Team 的 Room ID 映射
	h.initRoomIDMappings()
}

// initRoomIDMappings 从已有的 Team 数据初始化 Room ID 映射
// 这是为了让 MatrixBridge 知道 native room ID 对应的实际 Matrix room ID
func (h *OpenClawHandler) initRoomIDMappings() {
	if h.matrixBridge == nil {
		return
	}

	teams, err := h.loadTeams()
	if err != nil {
		log.Printf("openclaw: failed to load teams for room ID mapping: %v", err)
		return
	}

	for _, team := range teams {
		if team.MatrixRoomID != "" {
			// 使用 team ID 作为 native room ID 的基础
			// 因为 default room 的 ID 通常是 "team-{teamID}" 格式
			nativeRoomID := defaultRoomIDForTeam(team.ID)
			h.matrixBridge.RegisterRoomMapping(nativeRoomID, team.MatrixRoomID)
			log.Printf("openclaw: registered room mapping: %s -> %s", nativeRoomID, team.MatrixRoomID)
		}
	}
}

// GetMatrixBridge returns the Matrix bridge (may be nil if Matrix is disabled)
func (h *OpenClawHandler) GetMatrixBridge() *MatrixBridge {
	return h.matrixBridge
}

func (h *OpenClawHandler) SetRouterConfigPath(configPath string) {
	h.routerConfigPath = strings.TrimSpace(configPath)
}

func (h *OpenClawHandler) registryPath() string {
	return filepath.Join(h.dataDir, "containers.json")
}

func (h *OpenClawHandler) teamsPath() string {
	return filepath.Join(h.dataDir, "teams.json")
}

func (h *OpenClawHandler) roomsPath() string {
	return filepath.Join(h.dataDir, "rooms.json")
}

func (h *OpenClawHandler) roomMessagesPath(roomID string) string {
	return filepath.Join(h.dataDir, "room-messages", sanitizeRoomID(roomID)+".json")
}

func (h *OpenClawHandler) loadRegistry() ([]ContainerEntry, error) {
	data, err := os.ReadFile(h.registryPath())
	if err != nil {
		if os.IsNotExist(err) {
			return []ContainerEntry{}, nil
		}
		return nil, err
	}
	var entries []ContainerEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func (h *OpenClawHandler) saveRegistry(entries []ContainerEntry) error {
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(h.registryPath()), 0o755); err != nil {
		return err
	}
	return os.WriteFile(h.registryPath(), data, 0o644)
}

func (h *OpenClawHandler) loadTeams() ([]TeamEntry, error) {
	data, err := os.ReadFile(h.teamsPath())
	if err != nil {
		if os.IsNotExist(err) {
			return []TeamEntry{}, nil
		}
		return nil, err
	}
	var entries []TeamEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func (h *OpenClawHandler) saveTeams(entries []TeamEntry) error {
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(h.teamsPath()), 0o755); err != nil {
		return err
	}
	return os.WriteFile(h.teamsPath(), data, 0o644)
}

func findTeamByID(entries []TeamEntry, id string) *TeamEntry {
	// First pass: exact ID match
	for i := range entries {
		if entries[i].ID == id {
			return &entries[i]
		}
	}
	// Second pass: case-insensitive ID match
	lowerID := strings.ToLower(id)
	for i := range entries {
		if strings.ToLower(entries[i].ID) == lowerID {
			return &entries[i]
		}
	}
	// Third pass: ID prefix match (e.g., "vllm-sr" matches "vllm-sr-lab")
	for i := range entries {
		if strings.HasPrefix(strings.ToLower(entries[i].ID), lowerID) {
			return &entries[i]
		}
	}
	// Fourth pass: case-insensitive name match
	for i := range entries {
		if strings.EqualFold(entries[i].Name, id) {
			return &entries[i]
		}
	}
	return nil
}

func (h *OpenClawHandler) findEntry(name string) *ContainerEntry {
	entries, err := h.loadRegistry()
	if err != nil {
		return nil
	}
	for i := range entries {
		if entries[i].Name == name {
			return &entries[i]
		}
	}
	return nil
}

// defaultBridgeGatewayPort is the fixed port used for all OpenClaw containers
// when running in bridge network mode. Since each container has its own network
// namespace with a unique IP, port conflicts cannot occur.
const defaultBridgeGatewayPort = 18790

// isBridgeNetwork returns true if the network mode is a user-defined bridge network
// (not "host" and not "container:xxx"). In bridge mode, each container has an
// isolated network namespace, so all containers can safely bind to the same port.
func isBridgeNetwork(networkMode string) bool {
	nm := strings.ToLower(strings.TrimSpace(networkMode))
	if nm == "" || nm == "host" {
		return false
	}
	if strings.HasPrefix(nm, "container:") {
		return false
	}
	return true
}

func (h *OpenClawHandler) nextAvailablePort(networkMode string) int {
	// In bridge network mode, all containers can safely use the same port
	// because each container has its own network namespace with a unique IP.
	if isBridgeNetwork(networkMode) {
		return defaultBridgeGatewayPort
	}

	// Host network mode: need to find an available port on the host
	entries, _ := h.loadRegistry()
	used := map[int]bool{}
	for _, e := range entries {
		used[e.Port] = true
	}
	for port := 18788; ; port++ {
		if !used[port] && isTCPPortAvailable(port) {
			return port
		}
	}
}

func isTCPPortAvailable(port int) bool {
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return false
	}
	_ = ln.Close()
	return true
}

func canConnectTCP(host string, port int, timeout time.Duration) bool {
	addr := net.JoinHostPort(host, fmt.Sprintf("%d", port))
	conn, err := net.DialTimeout("tcp", addr, timeout)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

func detectContainerRuntime() (string, error) {
	candidates := []string{
		strings.TrimSpace(os.Getenv("OPENCLAW_CONTAINER_RUNTIME")),
		strings.TrimSpace(os.Getenv("CONTAINER_RUNTIME")),
		"docker",
		"podman",
		"/usr/local/bin/docker",
		"/usr/bin/docker",
		"/bin/docker",
		"/usr/local/bin/podman",
		"/usr/bin/podman",
		"/bin/podman",
	}

	seen := make(map[string]bool)
	checked := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate == "" || seen[candidate] {
			continue
		}
		seen[candidate] = true
		checked = append(checked, candidate)

		if filepath.IsAbs(candidate) {
			info, err := os.Stat(candidate)
			if err == nil && !info.IsDir() {
				return candidate, nil
			}
			continue
		}

		if resolved, err := exec.LookPath(candidate); err == nil {
			return resolved, nil
		}
	}

	return "", fmt.Errorf(
		"container runtime not available (checked: %s). PATH=%q. OpenClaw requires docker/podman in dashboard runtime. If you use `vllm-sr serve`, ensure vllm-sr image includes Docker CLI and mount /var/run/docker.sock",
		strings.Join(checked, ", "), os.Getenv("PATH"),
	)
}

func defaultOpenClawBaseImage() string {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_BASE_IMAGE")); candidate != "" {
		return candidate
	}
	// Default to openclaw-matrix which has Matrix plugin built-in.
	// This avoids the "keyed-async-queue" module error that occurs when
	// installing @openclaw/matrix at runtime on the official image.
	return "openclaw-matrix:latest"
}

func defaultOpenClawModelBaseURL() string {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_MODEL_BASE_URL")); candidate != "" {
		return candidate
	}
	return "http://127.0.0.1:8801/v1"
}

func (h *OpenClawHandler) resolveOpenClawModelBaseURL() string {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_MODEL_BASE_URL")); candidate != "" {
		return candidate
	}
	if candidate := h.discoverOpenClawModelBaseURLFromRouterConfig(); candidate != "" {
		return candidate
	}
	return defaultOpenClawModelBaseURL()
}

func (h *OpenClawHandler) discoverOpenClawModelBaseURLFromRouterConfig() string {
	configPath := strings.TrimSpace(h.routerConfigPath)
	if configPath == "" {
		return ""
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var config map[string]any
	if err := yaml.Unmarshal(data, &config); err != nil {
		return ""
	}

	for _, listener := range extractOpenClawRouterListeners(config) {
		port, ok := openClawToPort(listener["port"])
		if !ok {
			continue
		}
		host := formatOpenClawURLHost(normalizeOpenClawListenerHost(asString(listener["address"])))
		return fmt.Sprintf("http://%s:%d/v1", host, port)
	}

	return ""
}

func extractOpenClawRouterListeners(config map[string]any) []map[string]any {
	listeners := make([]map[string]any, 0)

	appendListeners := func(value any) {
		entries, ok := value.([]any)
		if !ok {
			return
		}
		for _, entry := range entries {
			if listener, ok := asStringMap(entry); ok {
				listeners = append(listeners, listener)
			}
		}
	}

	appendListeners(config["listeners"])

	if apiServer, ok := asStringMap(config["api_server"]); ok {
		appendListeners(apiServer["listeners"])
	}

	return listeners
}

func asStringMap(value any) (map[string]any, bool) {
	switch typed := value.(type) {
	case map[string]any:
		return typed, true
	case map[any]any:
		normalized := make(map[string]any, len(typed))
		for key, nested := range typed {
			textKey, ok := key.(string)
			if !ok {
				continue
			}
			normalized[textKey] = nested
		}
		return normalized, true
	default:
		return nil, false
	}
}

func asString(value any) string {
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func openClawToPort(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		if typed >= 1 && typed <= 65535 {
			return typed, true
		}
	case int64:
		port := int(typed)
		if port >= 1 && port <= 65535 {
			return port, true
		}
	case float64:
		port := int(typed)
		if port >= 1 && port <= 65535 && float64(port) == typed {
			return port, true
		}
	case string:
		port, err := strconv.Atoi(strings.TrimSpace(typed))
		if err == nil && port >= 1 && port <= 65535 {
			return port, true
		}
	}
	return 0, false
}

func normalizeOpenClawListenerHost(host string) string {
	if host == "" || host == "0.0.0.0" || host == "::" || host == "[::]" {
		return "127.0.0.1"
	}
	return host
}

func formatOpenClawURLHost(host string) string {
	if strings.Contains(host, ":") && !strings.HasPrefix(host, "[") && !strings.HasSuffix(host, "]") {
		return "[" + host + "]"
	}
	return host
}

func isContainerImageMissingError(output string) bool {
	lower := strings.ToLower(output)
	return strings.Contains(lower, "unable to find image") ||
		strings.Contains(lower, "pull access denied") ||
		strings.Contains(lower, "manifest unknown") ||
		strings.Contains(lower, "repository does not exist")
}

func (h *OpenClawHandler) imageExists(image string) bool {
	if strings.TrimSpace(image) == "" {
		return false
	}
	_, err := h.containerCombinedOutput("image", "inspect", image)
	return err == nil
}

func (h *OpenClawHandler) discoverLocalOpenClawImage() string {
	out, err := h.containerOutput("image", "ls", "--format", "{{.Repository}}:{{.Tag}}")
	if err != nil {
		return ""
	}

	seen := make(map[string]bool)
	latestCandidates := make([]string, 0)
	otherCandidates := make([]string, 0)
	for _, raw := range strings.Split(string(out), "\n") {
		image := strings.TrimSpace(raw)
		if image == "" || seen[image] {
			continue
		}
		seen[image] = true

		lower := strings.ToLower(image)
		if strings.Contains(lower, "<none>") {
			continue
		}
		if !strings.Contains(lower, "openclaw") {
			continue
		}
		if strings.HasSuffix(lower, ":latest") {
			latestCandidates = append(latestCandidates, image)
		} else {
			otherCandidates = append(otherCandidates, image)
		}
	}

	if len(latestCandidates) > 0 {
		return latestCandidates[0]
	}
	if len(otherCandidates) > 0 {
		return otherCandidates[0]
	}
	return ""
}

func (h *OpenClawHandler) resolveBaseImage(requested string) string {
	requested = strings.TrimSpace(requested)
	defaultImg := defaultOpenClawBaseImage() // openclaw-matrix:latest

	// If user explicitly requested a non-default image, use it
	if requested != "" && requested != defaultImg && requested != "ghcr.io/openclaw/openclaw:latest" {
		return requested
	}

	// Try the configured default (openclaw-matrix:latest)
	if h.imageExists(defaultImg) {
		return defaultImg
	}

	// Fallback: try to discover any local openclaw image
	discovered := h.discoverLocalOpenClawImage()
	if discovered != "" {
		log.Printf("openclaw: auto-selected local image %q (%s missing)", discovered, defaultImg)
		return discovered
	}

	// Last resort: return default and let ensureImageAvailable handle the error
	return defaultImg
}

func (h *OpenClawHandler) ensureImageAvailable(image string) error {
	image = strings.TrimSpace(image)
	if image == "" {
		return fmt.Errorf("OpenClaw image is empty")
	}
	if h.imageExists(image) {
		return nil
	}

	out, err := h.containerCombinedOutput("pull", image)
	if err == nil {
		log.Printf("openclaw: pulled missing image %q", image)
		return nil
	}

	trimmed := strings.TrimSpace(string(out))
	if strings.HasSuffix(strings.ToLower(image), ":local") {
		return fmt.Errorf(
			"OpenClaw image %q is missing locally and cannot be auto-pulled. Build/tag this image locally or set OPENCLAW_BASE_IMAGE to a pullable image",
			image,
		)
	}
	if trimmed == "" {
		return fmt.Errorf("failed to pull OpenClaw image %q", image)
	}
	return fmt.Errorf("failed to pull OpenClaw image %q: %s", image, trimmed)
}

func (h *OpenClawHandler) containerCommand(args ...string) (*exec.Cmd, error) {
	runtimeBin, err := detectContainerRuntime()
	if err != nil {
		return nil, err
	}
	return exec.Command(runtimeBin, args...), nil // #nosec G204
}

func (h *OpenClawHandler) containerOutput(args ...string) ([]byte, error) {
	cmd, err := h.containerCommand(args...)
	if err != nil {
		return nil, err
	}
	return cmd.Output()
}

func (h *OpenClawHandler) containerCombinedOutput(args ...string) ([]byte, error) {
	cmd, err := h.containerCommand(args...)
	if err != nil {
		return nil, err
	}
	return cmd.CombinedOutput()
}

func (h *OpenClawHandler) containerRun(args ...string) error {
	cmd, err := h.containerCommand(args...)
	if err != nil {
		return err
	}
	return cmd.Run()
}

// containerDataDir returns the per-container data directory.
func (h *OpenClawHandler) containerDataDir(name string) string {
	return filepath.Join(h.dataDir, "containers", name)
}

// --- Types ---

type SkillTemplate struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Emoji       string   `json:"emoji"`
	Category    string   `json:"category"`
	Builtin     bool     `json:"builtin"`
	Requires    []string `json:"requires,omitempty"`
	OS          []string `json:"os,omitempty"`
}

type IdentityConfig struct {
	Name       string `json:"name"`
	Emoji      string `json:"emoji"`
	Role       string `json:"role"`
	Vibe       string `json:"vibe"`
	Principles string `json:"principles"`
	Boundaries string `json:"boundaries"`
	UserName   string `json:"userName"`
	UserNotes  string `json:"userNotes"`
}

type ContainerConfig struct {
	ContainerName  string `json:"containerName"`
	GatewayPort    int    `json:"gatewayPort"`
	AuthToken      string `json:"authToken"`
	ModelBaseURL   string `json:"modelBaseUrl"`
	ModelAPIKey    string `json:"modelApiKey"`
	ModelName      string `json:"modelName"`
	MemoryBackend  string `json:"memoryBackend"`
	MemoryBaseURL  string `json:"memoryBaseUrl"`
	VectorStore    string `json:"vectorStore"`
	BrowserEnabled bool   `json:"browserEnabled"`
	BaseImage      string `json:"baseImage"`
	NetworkMode    string `json:"networkMode"`
	// Matrix communication configuration (injected by dashboard when MATRIX_ENABLED=true)
	MatrixEnabled     bool   `json:"matrixEnabled,omitempty"`
	MatrixHomeserver  string `json:"matrixHomeserver,omitempty"`
	MatrixDomain      string `json:"matrixDomain,omitempty"`
	MatrixAccessToken string `json:"matrixAccessToken,omitempty"`
	MatrixAdminUser   string `json:"matrixAdminUser,omitempty"`
}

type ProvisionRequest struct {
	Identity  IdentityConfig  `json:"identity"`
	Skills    []string        `json:"skills"`
	Container ContainerConfig `json:"container"`
	TeamID    string          `json:"teamId"`
	RoleKind  string          `json:"roleKind,omitempty"`
}

type ProvisionResponse struct {
	Success      bool   `json:"success"`
	Message      string `json:"message"`
	WorkspaceDir string `json:"workspaceDir,omitempty"`
	ConfigPath   string `json:"configPath,omitempty"`
	ContainerID  string `json:"containerId,omitempty"`
	DockerCmd    string `json:"dockerCmd,omitempty"`
	ComposeYAML  string `json:"composeYaml,omitempty"`
}

type OpenClawStatus struct {
	Running         bool   `json:"running"`
	ContainerName   string `json:"containerName,omitempty"`
	GatewayURL      string `json:"gatewayUrl,omitempty"`
	Port            int    `json:"port,omitempty"`
	Healthy         bool   `json:"healthy"`
	Error           string `json:"error,omitempty"`
	Image           string `json:"image,omitempty"`
	CreatedAt       string `json:"createdAt,omitempty"`
	TeamID          string `json:"teamId,omitempty"`
	TeamName        string `json:"teamName,omitempty"`
	AgentName       string `json:"agentName,omitempty"`
	AgentEmoji      string `json:"agentEmoji,omitempty"`
	AgentRole       string `json:"agentRole,omitempty"`
	AgentVibe       string `json:"agentVibe,omitempty"`
	AgentPrinciples string `json:"agentPrinciples,omitempty"`
	RoleKind        string `json:"roleKind,omitempty"`
}

type identitySnapshot struct {
	Name       string
	Emoji      string
	Role       string
	Vibe       string
	Principles string
}
