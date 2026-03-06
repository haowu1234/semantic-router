package handlers

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"
)

// CommunicationMode 定义通信模式
type CommunicationMode string

const (
	ModeNative CommunicationMode = "native"
	ModeMatrix CommunicationMode = "matrix"
	ModeHybrid CommunicationMode = "hybrid"
)

// MatrixBridgeConfig 桥接配置
type MatrixBridgeConfig struct {
	Mode           CommunicationMode
	ServerDomain   string
	InternalURL    string
	ExternalURL    string
	RegToken       string
	AdminUser      string
	SystemUser     string
	RoomModeMap    map[string]CommunicationMode
	SyncToMatrix   bool
	SyncFromMatrix bool
	DedupTTL       time.Duration
}

// MatrixBridge 通信桥接器
type MatrixBridge struct {
	config       MatrixBridgeConfig
	matrixClient *MatrixClient
	nativeStore  *NativeRoomStore
	dedupCache   *DedupCache
	roomModes    []roomModeRule
	mu           sync.RWMutex
}

type roomModeRule struct {
	pattern *regexp.Regexp
	mode    CommunicationMode
}

// NativeRoomStore 原生 Room 存储接口
type NativeRoomStore struct {
	handler *OpenClawHandler
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
func NewMatrixBridge(config MatrixBridgeConfig) (*MatrixBridge, error) {
	bridge := &MatrixBridge{
		config:     config,
		dedupCache: NewDedupCache(config.DedupTTL),
	}

	// 编译 Room 模式规则
	for pattern, mode := range config.RoomModeMap {
		re, err := compileGlobPattern(pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid room pattern %q: %w", pattern, err)
		}
		bridge.roomModes = append(bridge.roomModes, roomModeRule{
			pattern: re,
			mode:    mode,
		})
	}

	// 初始化 Matrix 客户端 (如果启用)
	if config.Mode == ModeMatrix || config.Mode == ModeHybrid {
		client, err := NewMatrixClient(MatrixClientConfig{
			HomeserverURL: config.InternalURL,
			Domain:        config.ServerDomain,
			SystemUser:    config.SystemUser,
			RegToken:      config.RegToken,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to init matrix client: %w", err)
		}
		bridge.matrixClient = client
	}

	return bridge, nil
}

// SetNativeStore 设置原生存储
func (b *MatrixBridge) SetNativeStore(store *NativeRoomStore) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.nativeStore = store
}

// GetRoomMode 获取 Room 的通信模式
func (b *MatrixBridge) GetRoomMode(roomID string) CommunicationMode {
	if b.config.Mode != ModeHybrid {
		return b.config.Mode
	}

	// 按顺序匹配规则
	for _, rule := range b.roomModes {
		if rule.pattern.MatchString(roomID) {
			return rule.mode
		}
	}
	return ModeNative
}

// SendMessage 发送消息 (自动路由到正确的后端)
func (b *MatrixBridge) SendMessage(ctx context.Context, msg *ClawRoomMessage) error {
	mode := b.GetRoomMode(msg.RoomID)

	// 检查去重
	if b.dedupCache.IsDuplicate(msg.ID) {
		return nil
	}
	b.dedupCache.Mark(msg.ID)

	switch mode {
	case ModeNative:
		return b.sendNative(ctx, msg)
	case ModeMatrix:
		return b.sendMatrix(ctx, msg)
	default:
		return fmt.Errorf("unknown mode: %s", mode)
	}
}

// sendNative 发送到原生 Room 系统
func (b *MatrixBridge) sendNative(ctx context.Context, msg *ClawRoomMessage) error {
	if b.nativeStore != nil {
		if err := b.nativeStore.SaveMessage(msg); err != nil {
			return err
		}
	}

	// 同步到 Matrix (如果启用)
	if b.config.SyncToMatrix && b.matrixClient != nil {
		go b.syncToMatrix(msg)
	}
	return nil
}

// sendMatrix 发送到 Matrix 服务器
func (b *MatrixBridge) sendMatrix(ctx context.Context, msg *ClawRoomMessage) error {
	if b.matrixClient == nil {
		return fmt.Errorf("matrix client not initialized")
	}

	matrixMsg := b.convertToMatrixMessage(msg)
	if err := b.matrixClient.SendMessage(ctx, matrixMsg); err != nil {
		return err
	}

	// 同步到 Native (如果启用)
	if b.config.SyncFromMatrix && b.nativeStore != nil {
		go b.syncFromMatrix(msg)
	}
	return nil
}

// syncToMatrix 同步消息到 Matrix
func (b *MatrixBridge) syncToMatrix(msg *ClawRoomMessage) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	matrixMsg := b.convertToMatrixMessage(msg)
	if err := b.matrixClient.SendMessage(ctx, matrixMsg); err != nil {
		// 记录错误但不阻塞
		fmt.Printf("failed to sync message to matrix: %v\n", err)
	}
}

// syncFromMatrix 同步消息到 Native
func (b *MatrixBridge) syncFromMatrix(msg *ClawRoomMessage) {
	if b.nativeStore != nil {
		if err := b.nativeStore.SaveMessage(msg); err != nil {
			fmt.Printf("failed to sync message from matrix: %v\n", err)
		}
	}
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

// GetMessages 获取房间消息
func (b *MatrixBridge) GetMessages(ctx context.Context, roomID string) ([]ClawRoomMessage, error) {
	mode := b.GetRoomMode(roomID)

	switch mode {
	case ModeNative:
		if b.nativeStore != nil {
			return b.nativeStore.GetMessages(roomID)
		}
		return nil, fmt.Errorf("native store not initialized")
	case ModeMatrix:
		if b.matrixClient != nil {
			return b.getMatrixMessages(ctx, roomID)
		}
		return nil, fmt.Errorf("matrix client not initialized")
	default:
		return nil, fmt.Errorf("unknown mode: %s", mode)
	}
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

// CreateRoom 创建房间
func (b *MatrixBridge) CreateRoom(ctx context.Context, name, teamID string, members []string) (string, error) {
	mode := b.GetRoomMode(fmt.Sprintf("team-%s", teamID))

	switch mode {
	case ModeMatrix:
		if b.matrixClient == nil {
			return "", fmt.Errorf("matrix client not initialized")
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

		return b.UnmapRoomID(matrixRoomID), nil

	default:
		// Native 模式不需要额外创建
		return fmt.Sprintf("team-%s", teamID), nil
	}
}

// NativeRoomStore 方法实现

// SaveMessage 保存消息到原生存储
func (s *NativeRoomStore) SaveMessage(msg *ClawRoomMessage) error {
	if s.handler == nil {
		return fmt.Errorf("handler not initialized")
	}

	s.handler.mu.Lock()
	defer s.handler.mu.Unlock()

	messages, err := s.handler.loadRoomMessages(msg.RoomID)
	if err != nil {
		return err
	}

	messages = append(messages, *msg)
	return s.handler.saveRoomMessages(msg.RoomID, messages)
}

// GetMessages 从原生存储获取消息
func (s *NativeRoomStore) GetMessages(roomID string) ([]ClawRoomMessage, error) {
	if s.handler == nil {
		return nil, fmt.Errorf("handler not initialized")
	}

	s.handler.mu.RLock()
	defer s.handler.mu.RUnlock()

	return s.handler.loadRoomMessages(roomID)
}

// 辅助函数: 编译 glob 模式为正则
func compileGlobPattern(pattern string) (*regexp.Regexp, error) {
	escaped := regexp.QuoteMeta(pattern)
	escaped = strings.ReplaceAll(escaped, `\*`, `.*`)
	escaped = strings.ReplaceAll(escaped, `\?`, `.`)
	return regexp.Compile("^" + escaped + "$")
}
