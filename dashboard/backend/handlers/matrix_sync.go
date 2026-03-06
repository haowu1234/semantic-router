package handlers

import (
	"context"
	"log"
	"sync"
	"time"
)

// MatrixSyncWorker Matrix 消息同步 Worker
type MatrixSyncWorker struct {
	client       *MatrixClient
	bridge       *MatrixBridge
	nativeHub    *WebSocketHub
	stopCh       chan struct{}
	nextBatch    string
	pollInterval time.Duration
	mu           sync.Mutex
	running      bool
}

// WebSocketHub WebSocket 广播接口 (需要在 openclaw_websocket.go 中实现)
type WebSocketHub interface {
	BroadcastToRoom(roomID string, message interface{})
}

// NewMatrixSyncWorker 创建同步 Worker
func NewMatrixSyncWorker(client *MatrixClient, bridge *MatrixBridge, hub *WebSocketHub) *MatrixSyncWorker {
	return &MatrixSyncWorker{
		client:       client,
		bridge:       bridge,
		nativeHub:    hub,
		stopCh:       make(chan struct{}),
		pollInterval: 30 * time.Second, // long-polling timeout
	}
}

// Start 启动同步
func (w *MatrixSyncWorker) Start(ctx context.Context) {
	w.mu.Lock()
	if w.running {
		w.mu.Unlock()
		return
	}
	w.running = true
	w.mu.Unlock()

	go w.syncLoop(ctx)
}

// Stop 停止同步
func (w *MatrixSyncWorker) Stop() {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.running {
		return
	}

	close(w.stopCh)
	w.running = false
}

// IsRunning 检查是否运行中
func (w *MatrixSyncWorker) IsRunning() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.running
}

// syncLoop 同步循环
func (w *MatrixSyncWorker) syncLoop(ctx context.Context) {
	log.Println("matrix sync worker started")
	defer log.Println("matrix sync worker stopped")

	// 首次同步，获取初始 token
	if err := w.initialSync(ctx); err != nil {
		log.Printf("matrix initial sync failed: %v", err)
	}

	for {
		select {
		case <-w.stopCh:
			return
		case <-ctx.Done():
			return
		default:
			if err := w.doSync(ctx); err != nil {
				log.Printf("matrix sync error: %v", err)
				// 错误后短暂等待再重试
				select {
				case <-time.After(5 * time.Second):
				case <-w.stopCh:
					return
				case <-ctx.Done():
					return
				}
			}
		}
	}
}

// initialSync 初始同步
func (w *MatrixSyncWorker) initialSync(ctx context.Context) error {
	syncResp, err := w.client.Sync(ctx, "", 0)
	if err != nil {
		return err
	}

	w.nextBatch = syncResp.NextBatch
	log.Printf("matrix initial sync complete, next_batch: %s", w.nextBatch)

	return nil
}

// doSync 执行一次同步
func (w *MatrixSyncWorker) doSync(ctx context.Context) error {
	syncResp, err := w.client.Sync(ctx, w.nextBatch, int(w.pollInterval.Milliseconds()))
	if err != nil {
		return err
	}

	w.nextBatch = syncResp.NextBatch

	// 处理新加入的房间邀请
	for roomID := range syncResp.Rooms.Invite {
		log.Printf("received room invite: %s", roomID)
		// 自动加入邀请的房间
		if err := w.client.JoinRoom(ctx, roomID); err != nil {
			log.Printf("failed to join invited room %s: %v", roomID, err)
		}
	}

	// 处理已加入房间的新消息
	for roomID, roomData := range syncResp.Rooms.Join {
		for _, event := range roomData.Timeline.Events {
			if event.Type == "m.room.message" {
				w.handleRoomMessage(roomID, &event)
			}
		}
	}

	return nil
}

// handleRoomMessage 处理 Matrix 房间消息
func (w *MatrixSyncWorker) handleRoomMessage(roomID string, event *MatrixEvent) {
	// 忽略自己发送的消息
	if event.Sender == w.client.GetUserID() {
		return
	}

	// 检查去重
	if w.bridge.dedupCache.IsDuplicate(event.EventID) {
		return
	}
	w.bridge.dedupCache.Mark(event.EventID)

	// 转换为 native 消息格式
	nativeRoomID := w.bridge.UnmapRoomID(roomID)
	nativeMsg := w.convertToNativeMessage(nativeRoomID, event)

	// 同步到 native 系统
	if w.bridge.config.SyncFromMatrix && w.bridge.nativeStore != nil {
		if err := w.bridge.nativeStore.SaveMessage(nativeMsg); err != nil {
			log.Printf("failed to sync matrix message to native: %v", err)
		}
	}

	// 广播到 WebSocket 客户端
	if w.nativeHub != nil {
		(*w.nativeHub).BroadcastToRoom(nativeMsg.RoomID, WSOutboundMessage{
			Type:    WSTypeNewMessage,
			RoomID:  nativeMsg.RoomID,
			Message: nativeMsg,
		})
	}

	log.Printf("synced message from matrix: room=%s, sender=%s", nativeRoomID, nativeMsg.SenderName)
}

// convertToNativeMessage 转换 Matrix 消息为 native 格式
func (w *MatrixSyncWorker) convertToNativeMessage(roomID string, event *MatrixEvent) *ClawRoomMessage {
	// 从元数据还原原始信息
	senderType := "user"
	senderID := ""
	senderName := w.bridge.UnmapUserID(event.Sender)
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

	// 提取 mentions
	var mentions []string
	if mentionsData, ok := event.Content["m.mentions"].(map[string]interface{}); ok {
		if userIDs, ok := mentionsData["user_ids"].([]interface{}); ok {
			for _, uid := range userIDs {
				if uidStr, ok := uid.(string); ok {
					mentions = append(mentions, w.bridge.UnmapUserID(uidStr))
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

// SetPollInterval 设置轮询间隔
func (w *MatrixSyncWorker) SetPollInterval(d time.Duration) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.pollInterval = d
}

// GetNextBatch 获取当前 sync token
func (w *MatrixSyncWorker) GetNextBatch() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.nextBatch
}
