package handlers

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// CommunicationMode 定义通信模式
// NOTE: 已移除 ModeNative 和 ModeHybrid，全部强制使用 Matrix 协议
type CommunicationMode string

const (
	// ModeMatrix 是唯一支持的通信模式，所有消息必须通过 Matrix 服务器
	// 如果 Matrix 服务器不可用，操作将失败而不是降级
	ModeMatrix CommunicationMode = "matrix"
)

// MatrixBridgeConfig 桥接配置
// NOTE: 已移除 SyncToMatrix/SyncFromMatrix，所有通信强制走 Matrix
type MatrixBridgeConfig struct {
	Mode         CommunicationMode
	ServerDomain string
	InternalURL  string
	ExternalURL  string
	RegToken     string
	AdminUser    string
	SystemUser   string
	DedupTTL     time.Duration
}

// MatrixBridge 通信桥接器
// NOTE: 已移除 nativeStore 和 roomModes，所有通信强制走 Matrix
type MatrixBridge struct {
	config       MatrixBridgeConfig
	matrixClient *MatrixClient
	dedupCache   *DedupCache
	mu           sync.RWMutex
}

// DedupCache 消息去重缓存
type DedupCache struct {
	cache map[string]time.Time
	ttl   time.Duration
	mu    sync.RWMutex
}

// NewDedupCache 创建去重缓存
func NewDedupCache(ttl time.Duration) *DedupCache {
	cache := &DedupCache{
		cache: make(map[string]time.Time),
		ttl:   ttl,
	}
	go cache.cleanup()
	return cache
}

// IsDuplicate 检查是否重复
func (c *DedupCache) IsDuplicate(id string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	_, exists := c.cache[id]
	return exists
}

// Mark 标记消息
func (c *DedupCache) Mark(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache[id] = time.Now()
}

// cleanup 定期清理过期条目
func (c *DedupCache) cleanup() {
	ticker := time.NewTicker(c.ttl)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		for id, ts := range c.cache {
			if now.Sub(ts) > c.ttl {
				delete(c.cache, id)
			}
		}
		c.mu.Unlock()
	}
}

// NewMatrixBridge 创建通信桥接器
// NOTE: Matrix 客户端是必须的，如果初始化失败则返回错误
func NewMatrixBridge(config MatrixBridgeConfig) (*MatrixBridge, error) {
	// 强制使用 Matrix 模式
	config.Mode = ModeMatrix

	bridge := &MatrixBridge{
		config:     config,
		dedupCache: NewDedupCache(config.DedupTTL),
	}

	// Matrix 客户端是必须的，不再可选
	client, err := NewMatrixClient(MatrixClientConfig{
		HomeserverURL: config.InternalURL,
		Domain:        config.ServerDomain,
		SystemUser:    config.SystemUser,
		RegToken:      config.RegToken,
	})
	if err != nil {
		return nil, fmt.Errorf("matrix client is required but failed to initialize: %w", err)
	}
	bridge.matrixClient = client

	return bridge, nil
}

// GetRoomMode 获取 Room 的通信模式
// NOTE: 始终返回 ModeMatrix，不再支持其他模式
func (b *MatrixBridge) GetRoomMode(roomID string) CommunicationMode {
	return ModeMatrix
}

// SendMessage 发送消息到 Matrix 服务器
// NOTE: 不再支持 fallback 到本地存储，Matrix 不可用时返回错误
func (b *MatrixBridge) SendMessage(ctx context.Context, msg *ClawRoomMessage) error {
	// 检查去重
	if b.dedupCache.IsDuplicate(msg.ID) {
		return nil
	}
	b.dedupCache.Mark(msg.ID)

	// 直接发送到 Matrix，不再有 fallback
	return b.sendMatrix(ctx, msg)
}

// sendMatrix 发送到 Matrix 服务器
// NOTE: 这是唯一的发送方式，不再有 fallback
func (b *MatrixBridge) sendMatrix(ctx context.Context, msg *ClawRoomMessage) error {
	if b.matrixClient == nil {
		return fmt.Errorf("matrix client not initialized - matrix is required for all communication")
	}

	matrixMsg := b.convertToMatrixMessage(msg)
	return b.matrixClient.SendMessage(ctx, matrixMsg)
}

// convertToMatrixMessage 转换消息格式
func (b *MatrixBridge) convertToMatrixMessage(msg *ClawRoomMessage) *MatrixMessage {
	matrixRoomID := b.MapRoomID(msg.RoomID)

	// 构建 m.mentions
	mentions := &MatrixMentions{}
	for _, mention := range msg.Mentions {
		userID := b.MapUserID(mention)
		mentions.UserIDs = append(mentions.UserIDs, userID)
	}

	return &MatrixMessage{
		RoomID:   matrixRoomID,
		MsgType:  "m.text",
		Body:     msg.Content,
		Mentions: mentions,
		Metadata: map[string]interface{}{
			"semantic_router.sender_type": msg.SenderType,
			"semantic_router.sender_id":   msg.SenderID,
			"semantic_router.sender_name": msg.SenderName,
			"semantic_router.room_id":     msg.RoomID,
			"semantic_router.team_id":     msg.TeamID,
		},
	}
}

// MapRoomID 映射 Room ID (native → Matrix)
func (b *MatrixBridge) MapRoomID(nativeID string) string {
	// 格式: !<room_id>:<domain>
	return fmt.Sprintf("!%s:%s", nativeID, b.config.ServerDomain)
}

// UnmapRoomID 反向映射 Room ID (Matrix → native)
func (b *MatrixBridge) UnmapRoomID(matrixID string) string {
	// 格式: !<room_id>:<domain> → <room_id>
	if !strings.HasPrefix(matrixID, "!") {
		return matrixID
	}
	parts := strings.SplitN(matrixID[1:], ":", 2)
	if len(parts) == 0 {
		return matrixID
	}
	return parts[0]
}

// MapUserID 映射 User ID (native → Matrix)
func (b *MatrixBridge) MapUserID(nativeID string) string {
	// 格式: @<user_id>:<domain>
	if strings.HasPrefix(nativeID, "@") {
		return nativeID // 已经是 Matrix 格式
	}
	return fmt.Sprintf("@%s:%s", nativeID, b.config.ServerDomain)
}

// UnmapUserID 反向映射 User ID (Matrix → native)
func (b *MatrixBridge) UnmapUserID(matrixID string) string {
	// 格式: @<user_id>:<domain> → <user_id>
	if !strings.HasPrefix(matrixID, "@") {
		return matrixID
	}
	parts := strings.SplitN(matrixID[1:], ":", 2)
	if len(parts) == 0 {
		return matrixID
	}
	return parts[0]
}

// GetMessages 从 Matrix 获取房间消息
// NOTE: 不再支持从本地存储获取，只从 Matrix 获取
func (b *MatrixBridge) GetMessages(ctx context.Context, roomID string) ([]ClawRoomMessage, error) {
	if b.matrixClient == nil {
		return nil, fmt.Errorf("matrix client not initialized - matrix is required for all communication")
	}
	return b.getMatrixMessages(ctx, roomID)
}

// getMatrixMessages 从 Matrix 获取消息
func (b *MatrixBridge) getMatrixMessages(ctx context.Context, roomID string) ([]ClawRoomMessage, error) {
	matrixRoomID := b.MapRoomID(roomID)
	events, err := b.matrixClient.GetRoomMessages(ctx, matrixRoomID, 100)
	if err != nil {
		return nil, err
	}

	var messages []ClawRoomMessage
	for _, event := range events {
		msg := b.convertFromMatrixEvent(roomID, &event)
		messages = append(messages, *msg)
	}
	return messages, nil
}

// convertFromMatrixEvent 从 Matrix 事件转换
func (b *MatrixBridge) convertFromMatrixEvent(roomID string, event *MatrixEvent) *ClawRoomMessage {
	senderType := "user"
	senderID := ""
	senderName := b.UnmapUserID(event.Sender)
	teamID := ""

	if meta, ok := event.Content["semantic_router.sender_type"].(string); ok {
		senderType = meta
	}
	if meta, ok := event.Content["semantic_router.sender_id"].(string); ok {
		senderID = meta
	}
	if meta, ok := event.Content["semantic_router.sender_name"].(string); ok {
		senderName = meta
	}
	if meta, ok := event.Content["semantic_router.team_id"].(string); ok {
		teamID = meta
	}

	var mentions []string
	if mentionsData, ok := event.Content["m.mentions"].(map[string]interface{}); ok {
		if userIDs, ok := mentionsData["user_ids"].([]interface{}); ok {
			for _, uid := range userIDs {
				if uidStr, ok := uid.(string); ok {
					mentions = append(mentions, b.UnmapUserID(uidStr))
				}
			}
		}
	}

	body := ""
	if bodyStr, ok := event.Content["body"].(string); ok {
		body = bodyStr
	}

	return &ClawRoomMessage{
		ID:         event.EventID,
		RoomID:     roomID,
		TeamID:     teamID,
		SenderType: senderType,
		SenderID:   senderID,
		SenderName: senderName,
		Content:    body,
		Mentions:   mentions,
		CreatedAt:  time.UnixMilli(event.OriginServerTS).Format(time.RFC3339),
	}
}

// CreateRoom 在 Matrix 服务器上创建房间
// NOTE: 不再支持 Native 模式，房间必须在 Matrix 上创建
// Returns the full Matrix room ID (e.g., !abc123:matrix.domain) for storage
func (b *MatrixBridge) CreateRoom(ctx context.Context, name, teamID string, members []string) (string, error) {
	if b.matrixClient == nil {
		return "", fmt.Errorf("matrix client not initialized - matrix is required for all communication")
	}

	// 转换成员 ID
	var matrixMembers []string
	for _, m := range members {
		matrixMembers = append(matrixMembers, b.MapUserID(m))
	}

	matrixRoomID, err := b.matrixClient.CreateRoom(ctx, &CreateRoomRequest{
		Name:   name,
		Topic:  fmt.Sprintf("Semantic Router Team Room: %s", teamID),
		Invite: matrixMembers,
	})
	if err != nil {
		return "", err
	}

	// Return the full Matrix room ID (e.g., !abc123:domain) instead of unmapped ID
	// This is needed for InviteUser and other Matrix API calls
	return matrixRoomID, nil
}

// NOTE: NativeRoomStore 已移除，不再支持本地存储
// 所有消息存储和检索必须通过 Matrix 服务器
