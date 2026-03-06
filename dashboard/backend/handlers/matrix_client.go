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
	HomeserverURL string
	Domain        string
	SystemUser    string
	RegToken      string
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
			Timeout: 30 * time.Second,
		},
	}

	// 登录系统账户
	if err := client.login(); err != nil {
		return nil, fmt.Errorf("matrix login failed: %w", err)
	}

	return client, nil
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
	payload := map[string]interface{}{
		"name":       req.Name,
		"topic":      req.Topic,
		"invite":     req.Invite,
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

// RegisterUser 注册新用户 (需要 admin 权限)
func (c *MatrixClient) RegisterUser(ctx context.Context, username, password string) (string, error) {
	regReq := map[string]interface{}{
		"username":                    username,
		"password":                    password,
		"registration_token":          c.config.RegToken,
		"device_id":                   fmt.Sprintf("semantic-router-%s", username),
		"initial_device_display_name": fmt.Sprintf("Semantic Router %s", username),
	}

	resp, err := c.doRequest("POST", "/_matrix/client/v3/register", regReq)
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

	return regResp.UserID, nil
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

// Close 关闭客户端
func (c *MatrixClient) Close() error {
	// 可以选择登出
	_, _ = c.doRequestWithAuth("POST", "/_matrix/client/v3/logout", nil)
	return nil
}
