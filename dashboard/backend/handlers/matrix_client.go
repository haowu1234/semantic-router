package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// MatrixClientConfig 客户端配置
type MatrixClientConfig struct {
	HomeserverURL     string
	Domain            string
	SystemUser        string
	RegToken          string
	SystemAccessToken string // 可选：如果提供，则直接使用此 token，跳过密码登录
}

// MatrixError 表示 Matrix API 返回的错误
type MatrixError struct {
	StatusCode int
	ErrCode    string
	Message    string
	Body       string
}

func (e *MatrixError) Error() string {
	if e.ErrCode != "" {
		return fmt.Sprintf("matrix error %s: %s", e.ErrCode, e.Message)
	}
	return fmt.Sprintf("matrix api error %d: %s", e.StatusCode, e.Body)
}

// MatrixClient Matrix 客户端
type MatrixClient struct {
	config      MatrixClientConfig
	httpClient  *http.Client
	accessToken string
	userID      string
	txnID       int64
	mu          sync.Mutex
}

// MatrixMessage Matrix 消息
type MatrixMessage struct {
	RoomID   string                 `json:"-"`
	MsgType  string                 `json:"msgtype"`
	Body     string                 `json:"body"`
	Mentions *MatrixMentions        `json:"m.mentions,omitempty"`
	Metadata map[string]interface{} `json:"-"`
}

// MatrixMentions 提及信息 (MSC3952)
type MatrixMentions struct {
	UserIDs []string `json:"user_ids,omitempty"`
	Room    bool     `json:"room,omitempty"`
}

// CreateRoomRequest 创建房间请求
type CreateRoomRequest struct {
	Name     string
	Topic    string
	Invite   []string
	IsDirect bool
}

// MatrixEvent Matrix 事件
type MatrixEvent struct {
	Type           string                 `json:"type"`
	EventID        string                 `json:"event_id"`
	Sender         string                 `json:"sender"`
	OriginServerTS int64                  `json:"origin_server_ts"`
	Content        map[string]interface{} `json:"content"`
}

// MatrixSyncResponse Matrix sync 响应
type MatrixSyncResponse struct {
	NextBatch string `json:"next_batch"`
	Rooms     struct {
		Join map[string]struct {
			Timeline struct {
				Events    []MatrixEvent `json:"events"`
				PrevBatch string        `json:"prev_batch"`
			} `json:"timeline"`
			State struct {
				Events []MatrixEvent `json:"events"`
			} `json:"state"`
		} `json:"join"`
		Invite map[string]struct {
			InviteState struct {
				Events []MatrixEvent `json:"events"`
			} `json:"invite_state"`
		} `json:"invite"`
	} `json:"rooms"`
}

// MatrixMessagesResponse 消息列表响应
type MatrixMessagesResponse struct {
	Start string        `json:"start"`
	End   string        `json:"end"`
	Chunk []MatrixEvent `json:"chunk"`
}

// NewMatrixClient 创建客户端
func NewMatrixClient(config MatrixClientConfig) (*MatrixClient, error) {
	client := &MatrixClient{
		config: config,
		httpClient: &http.Client{
			Timeout: 60 * time.Second, // 需要比 sync timeout (30s) 更长，留出网络传输时间
		},
	}

	// 如果提供了 SystemAccessToken，直接使用，跳过密码登录
	if config.SystemAccessToken != "" {
		if err := client.validateAndUseToken(); err != nil {
			return nil, fmt.Errorf("matrix token validation failed: %w", err)
		}
		return client, nil
	}

	// 否则使用密码登录
	if err := client.login(); err != nil {
		return nil, fmt.Errorf("matrix login failed: %w", err)
	}

	return client, nil
}

// validateAndUseToken 验证并使用预配置的 access token
func (c *MatrixClient) validateAndUseToken() error {
	c.accessToken = c.config.SystemAccessToken

	// 调用 whoami 验证 token 有效性（必须使用带 token 的请求）
	resp, err := c.doRequestWithAuth("GET", "/_matrix/client/v3/account/whoami", nil)
	if err != nil {
		return fmt.Errorf("token validation failed: %w", err)
	}

	var whoamiResp struct {
		UserID   string `json:"user_id"`
		DeviceID string `json:"device_id"`
	}
	if err := json.Unmarshal(resp, &whoamiResp); err != nil {
		return fmt.Errorf("failed to parse whoami response: %w", err)
	}

	c.userID = whoamiResp.UserID
	return nil
}

// login 登录 Matrix 服务器
func (c *MatrixClient) login() error {
	// 首先尝试注册 (幂等)
	_ = c.register() // 忽略注册错误，可能已存在

	loginReq := map[string]interface{}{
		"type": "m.login.password",
		"identifier": map[string]interface{}{
			"type": "m.id.user",
			"user": c.config.SystemUser,
		},
		"password":                    c.config.RegToken,
		"device_id":                   "semantic-router-dashboard",
		"initial_device_display_name": "Semantic Router Dashboard",
	}

	resp, err := c.doRequest("POST", "/_matrix/client/v3/login", loginReq)
	if err != nil {
		return err
	}

	var loginResp struct {
		AccessToken string `json:"access_token"`
		UserID      string `json:"user_id"`
		DeviceID    string `json:"device_id"`
	}
	if err := json.Unmarshal(resp, &loginResp); err != nil {
		return err
	}

	c.accessToken = loginResp.AccessToken
	c.userID = loginResp.UserID
	return nil
}

// register 注册系统账户
func (c *MatrixClient) register() error {
	regReq := map[string]interface{}{
		"username":                    c.config.SystemUser,
		"password":                    c.config.RegToken,
		"registration_token":          c.config.RegToken,
		"device_id":                   "semantic-router-dashboard",
		"initial_device_display_name": "Semantic Router Dashboard",
	}

	_, err := c.doRequest("POST", "/_matrix/client/v3/register", regReq)
	return err
}

// GetUserID 获取当前用户 ID
func (c *MatrixClient) GetUserID() string {
	return c.userID
}

// SendMessage 发送消息
func (c *MatrixClient) SendMessage(ctx context.Context, msg *MatrixMessage) error {
	c.mu.Lock()
	c.txnID++
	txnID := c.txnID
	c.mu.Unlock()

	endpoint := fmt.Sprintf("/_matrix/client/v3/rooms/%s/send/m.room.message/%d",
		msg.RoomID, txnID)

	payload := map[string]interface{}{
		"msgtype": msg.MsgType,
		"body":    msg.Body,
	}

	// 添加 m.mentions (关键！Worker 只响应被正确 @ 的消息)
	if msg.Mentions != nil && (len(msg.Mentions.UserIDs) > 0 || msg.Mentions.Room) {
		payload["m.mentions"] = msg.Mentions
	}

	// 添加自定义元数据
	for k, v := range msg.Metadata {
		payload[k] = v
	}

	_, err := c.doRequestWithAuth("PUT", endpoint, payload)
	return err
}

// CreateRoom 创建房间
func (c *MatrixClient) CreateRoom(ctx context.Context, req *CreateRoomRequest) (string, error) {
	// 确保 invite 是空数组而不是 null，Matrix API 要求必须是数组类型
	invite := req.Invite
	if invite == nil {
		invite = []string{}
	}

	payload := map[string]interface{}{
		"name":       req.Name,
		"topic":      req.Topic,
		"invite":     invite,
		"preset":     "trusted_private_chat",
		"visibility": "private",
	}

	if req.IsDirect {
		payload["is_direct"] = true
	}

	resp, err := c.doRequestWithAuth("POST", "/_matrix/client/v3/createRoom", payload)
	if err != nil {
		return "", err
	}

	var createResp struct {
		RoomID string `json:"room_id"`
	}
	if err := json.Unmarshal(resp, &createResp); err != nil {
		return "", err
	}

	return createResp.RoomID, nil
}

// JoinRoom 加入房间
func (c *MatrixClient) JoinRoom(ctx context.Context, roomID string) error {
	_, err := c.doRequestWithAuth("POST",
		fmt.Sprintf("/_matrix/client/v3/rooms/%s/join", roomID), nil)
	return err
}

// LeaveRoom 离开房间
func (c *MatrixClient) LeaveRoom(ctx context.Context, roomID string) error {
	_, err := c.doRequestWithAuth("POST",
		fmt.Sprintf("/_matrix/client/v3/rooms/%s/leave", roomID), nil)
	return err
}

// InviteUser 邀请用户
func (c *MatrixClient) InviteUser(ctx context.Context, roomID, userID string) error {
	payload := map[string]interface{}{
		"user_id": userID,
	}
	_, err := c.doRequestWithAuth("POST",
		fmt.Sprintf("/_matrix/client/v3/rooms/%s/invite", roomID), payload)
	return err
}

// KickUser 踢出用户
func (c *MatrixClient) KickUser(ctx context.Context, roomID, userID, reason string) error {
	payload := map[string]interface{}{
		"user_id": userID,
		"reason":  reason,
	}
	_, err := c.doRequestWithAuth("POST",
		fmt.Sprintf("/_matrix/client/v3/rooms/%s/kick", roomID), payload)
	return err
}

// GetRoomMessages 获取房间消息
func (c *MatrixClient) GetRoomMessages(ctx context.Context, roomID string, limit int) ([]MatrixEvent, error) {
	endpoint := fmt.Sprintf("/_matrix/client/v3/rooms/%s/messages?dir=b&limit=%d",
		roomID, limit)

	resp, err := c.doRequestWithAuth("GET", endpoint, nil)
	if err != nil {
		return nil, err
	}

	var messagesResp MatrixMessagesResponse
	if err := json.Unmarshal(resp, &messagesResp); err != nil {
		return nil, err
	}

	// 过滤出 m.room.message 事件
	var messages []MatrixEvent
	for _, event := range messagesResp.Chunk {
		if event.Type == "m.room.message" {
			messages = append(messages, event)
		}
	}

	return messages, nil
}

// GetJoinedRooms 获取已加入的房间
func (c *MatrixClient) GetJoinedRooms(ctx context.Context) ([]string, error) {
	resp, err := c.doRequestWithAuth("GET", "/_matrix/client/v3/joined_rooms", nil)
	if err != nil {
		return nil, err
	}

	var roomsResp struct {
		JoinedRooms []string `json:"joined_rooms"`
	}
	if err := json.Unmarshal(resp, &roomsResp); err != nil {
		return nil, err
	}

	return roomsResp.JoinedRooms, nil
}

// SetRoomTopic 设置房间主题
func (c *MatrixClient) SetRoomTopic(ctx context.Context, roomID, topic string) error {
	payload := map[string]interface{}{
		"topic": topic,
	}
	_, err := c.doRequestWithAuth("PUT",
		fmt.Sprintf("/_matrix/client/v3/rooms/%s/state/m.room.topic", roomID), payload)
	return err
}

// SetRoomName 设置房间名称
func (c *MatrixClient) SetRoomName(ctx context.Context, roomID, name string) error {
	payload := map[string]interface{}{
		"name": name,
	}
	_, err := c.doRequestWithAuth("PUT",
		fmt.Sprintf("/_matrix/client/v3/rooms/%s/state/m.room.name", roomID), payload)
	return err
}

// Sync 执行 Matrix sync (用于获取新消息)
func (c *MatrixClient) Sync(ctx context.Context, since string, timeout int) (*MatrixSyncResponse, error) {
	endpoint := fmt.Sprintf("/_matrix/client/v3/sync?timeout=%d", timeout)
	if since != "" {
		endpoint += "&since=" + since
	}

	resp, err := c.doRequestWithAuth("GET", endpoint, nil)
	if err != nil {
		return nil, err
	}

	var syncResp MatrixSyncResponse
	if err := json.Unmarshal(resp, &syncResp); err != nil {
		return nil, err
	}

	return &syncResp, nil
}

// RegisterUser 注册新用户并返回 access token (使用 registration token)
// Matrix 使用 User-Interactive Authentication (UIA) 流程:
// 1. 首次请求返回 401 + session + flows
// 2. 使用 session 和 auth 参数完成认证阶段
func (c *MatrixClient) RegisterUser(ctx context.Context, username, password string) (string, error) {
	// Step 1: 初始请求获取 session
	initReq := map[string]interface{}{
		"username":                    username,
		"password":                    password,
		"device_id":                   fmt.Sprintf("semantic-router-%s", username),
		"initial_device_display_name": fmt.Sprintf("Semantic Router %s", username),
	}

	resp, err := c.doRequestRaw("POST", "/_matrix/client/v3/register", initReq)
	if err == nil {
		// 直接成功（服务器可能不需要 UIA）
		var regResp struct {
			UserID      string `json:"user_id"`
			AccessToken string `json:"access_token"`
		}
		if err := json.Unmarshal(resp, &regResp); err != nil {
			return "", err
		}
		return regResp.AccessToken, nil
	}

	// Step 2: 解析 UIA 响应获取 session
	matrixErr, ok := err.(*MatrixError)
	if !ok || matrixErr.StatusCode != 401 {
		return "", err
	}

	var uiaResp struct {
		Session string `json:"session"`
		Flows   []struct {
			Stages []string `json:"stages"`
		} `json:"flows"`
	}
	if parseErr := json.Unmarshal([]byte(matrixErr.Body), &uiaResp); parseErr != nil {
		return "", fmt.Errorf("failed to parse UIA response: %w", parseErr)
	}

	if uiaResp.Session == "" {
		return "", fmt.Errorf("matrix UIA response missing session")
	}

	// Step 3: 完成 registration_token 认证阶段
	authReq := map[string]interface{}{
		"username":                    username,
		"password":                    password,
		"device_id":                   fmt.Sprintf("semantic-router-%s", username),
		"initial_device_display_name": fmt.Sprintf("Semantic Router %s", username),
		"auth": map[string]interface{}{
			"type":    "m.login.registration_token",
			"token":   c.config.RegToken,
			"session": uiaResp.Session,
		},
	}

	resp, err = c.doRequest("POST", "/_matrix/client/v3/register", authReq)
	if err != nil {
		return "", err
	}

	var regResp struct {
		UserID      string `json:"user_id"`
		AccessToken string `json:"access_token"`
	}
	if err := json.Unmarshal(resp, &regResp); err != nil {
		return "", err
	}

	return regResp.AccessToken, nil
}

// LoginUser 登录用户并返回 access token
func (c *MatrixClient) LoginUser(ctx context.Context, username, password string) (string, error) {
	loginReq := map[string]interface{}{
		"type": "m.login.password",
		"identifier": map[string]interface{}{
			"type": "m.id.user",
			"user": username,
		},
		"password":                    password,
		"device_id":                   fmt.Sprintf("semantic-router-%s", username),
		"initial_device_display_name": fmt.Sprintf("Semantic Router %s", username),
	}

	resp, err := c.doRequest("POST", "/_matrix/client/v3/login", loginReq)
	if err != nil {
		return "", err
	}

	var loginResp struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.Unmarshal(resp, &loginResp); err != nil {
		return "", err
	}

	return loginResp.AccessToken, nil
}

// doRequest 发送 HTTP 请求
func (c *MatrixClient) doRequest(method, endpoint string, body interface{}) ([]byte, error) {
	return c.doRequestWithToken(method, endpoint, body, "")
}

// doRequestRaw 发送 HTTP 请求，返回 MatrixError 而不是简单的 error（用于 UIA 流程）
func (c *MatrixClient) doRequestRaw(method, endpoint string, body interface{}) ([]byte, error) {
	return c.doRequestWithTokenRaw(method, endpoint, body, "")
}

// doRequestWithAuth 发送带认证的请求
func (c *MatrixClient) doRequestWithAuth(method, endpoint string, body interface{}) ([]byte, error) {
	return c.doRequestWithToken(method, endpoint, body, c.accessToken)
}

func (c *MatrixClient) doRequestWithToken(method, endpoint string, body interface{}, token string) ([]byte, error) {
	url := c.config.HomeserverURL + endpoint

	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequest(method, url, bodyReader)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		// 解析 Matrix 错误响应
		var matrixErr struct {
			ErrCode string `json:"errcode"`
			Error   string `json:"error"`
		}
		if err := json.Unmarshal(respBody, &matrixErr); err == nil && matrixErr.ErrCode != "" {
			return nil, fmt.Errorf("matrix error %s: %s", matrixErr.ErrCode, matrixErr.Error)
		}
		return nil, fmt.Errorf("matrix api error %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// doRequestWithTokenRaw 发送请求，返回 MatrixError 结构（用于 UIA 流程）
func (c *MatrixClient) doRequestWithTokenRaw(method, endpoint string, body interface{}, token string) ([]byte, error) {
	url := c.config.HomeserverURL + endpoint

	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequest(method, url, bodyReader)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		// 返回 MatrixError 以便调用者可以解析完整响应
		var parsed struct {
			ErrCode string `json:"errcode"`
			Error   string `json:"error"`
		}
		_ = json.Unmarshal(respBody, &parsed)
		return nil, &MatrixError{
			StatusCode: resp.StatusCode,
			ErrCode:    parsed.ErrCode,
			Message:    parsed.Error,
			Body:       string(respBody),
		}
	}

	return respBody, nil
}

// Close 关闭客户端
func (c *MatrixClient) Close() error {
	// 可以选择登出
	_, _ = c.doRequestWithAuth("POST", "/_matrix/client/v3/logout", nil)
	return nil
}
