# Dashboard 只读模式设计方案

> **Issue**: [#1110 - Add --readonly-dashboard flag to vllm-sr serve for public beta deployments](https://github.com/vllm-project/semantic-router/issues/1110)

## 1. 需求背景

### 1.1 业务场景

在公开 Beta 测试环境中，需要限制用户的操作权限，允许用户体验 Playground 对话功能，但禁止修改任何配置。

| 功能 | 正常模式 | 只读模式 |
|------|:-------:|:-------:|
| Playground 对话 | ✅ | ✅ |
| 查看所有配置 | ✅ | ✅ |
| 修改模型配置 (Models) | ✅ | ❌ |
| 修改端点设置 (Endpoints) | ✅ | ❌ |
| 修改 Prompt Guard 配置 | ✅ | ❌ |
| 修改 Semantic Cache 配置 | ✅ | ❌ |
| 修改 Categories / Reasoning Families | ✅ | ❌ |
| 修改 Tools 配置 | ✅ | ❌ |
| 修改 Observability 设置 | ✅ | ❌ |
| 修改 Batch Classification API | ✅ | ❌ |

### 1.2 使用方式

```bash
# 正常模式 (完整编辑权限)
vllm-sr serve --config config.yaml

# 只读模式 (公开 Beta 环境)
vllm-sr serve --config config.yaml --readonly-dashboard
```

---

## 2. 整体架构

### 2.1 数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  vllm-sr serve --readonly-dashboard                                      │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────┐    DASHBOARD_READONLY=true    ┌─────────────────┐   │
│  │  CLI (Python)   │ ─────────────────────────────▶│ Docker Container │   │
│  │  main.py        │         环境变量               │                 │   │
│  └─────────────────┘                               └────────┬────────┘   │
│                                                             │            │
│                                                             ▼            │
│                                                   ┌─────────────────┐    │
│                                                   │ start-dashboard │    │
│                                                   │     .sh         │    │
│                                                   └────────┬────────┘    │
│                                                             │            │
│                                                             │ -readonly  │
│                                                             ▼            │
│  ┌─────────────────┐    GET /api/settings    ┌─────────────────┐        │
│  │  Frontend       │◀────────────────────────│  Go Backend     │        │
│  │  (React)        │    {readonlyMode: true} │                 │        │
│  │                 │                         │  config.go      │        │
│  │  - 隐藏编辑按钮  │───POST /api/config────▶│  handlers/*.go  │        │
│  │  - 显示只读提示  │                         │  返回 403       │        │
│  └─────────────────┘                         └─────────────────┘        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 修改文件清单

| 层级 | 文件 | 修改内容 |
|------|------|----------|
| CLI | `src/vllm-sr/cli/main.py` | 添加 `--readonly-dashboard` 参数 |
| Core | `src/vllm-sr/cli/core.py` | 传递环境变量 `DASHBOARD_READONLY` |
| Shell | `src/vllm-sr/start-dashboard.sh` | 解析环境变量，传递 `-readonly` 给后端 |
| Backend Config | `dashboard/backend/config/config.go` | 新增 `ReadonlyMode` 字段 |
| Backend Main | `dashboard/backend/main.go` | 传递 `ReadonlyMode` 给 handlers |
| Backend Handlers | `dashboard/backend/handlers/config.go` | 写操作检查只读模式，返回 403 |
| Frontend Context | `dashboard/frontend/src/contexts/ReadonlyContext.tsx` | 创建全局只读状态 (新建) |
| Frontend Component | `dashboard/frontend/src/components/ReadonlyBanner.tsx` | 只读提示 Banner (新建) |
| Frontend Pages | `dashboard/frontend/src/pages/ConfigPage.tsx` | 隐藏/禁用编辑按钮 |
| Documentation | `dashboard/README.md` | 更新安全与访问控制章节 |

---

## 3. 详细设计

### 3.1 CLI 层 (Python)

#### 3.1.1 src/vllm-sr/cli/main.py

```python
@click.command()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.option("--image", default=None, help="Docker image to use")
@click.option(
    "--readonly-dashboard",
    is_flag=True,
    default=False,
    help="Run dashboard in read-only mode (disable config editing, allow playground only)"
)
def serve(config, image, ..., readonly_dashboard):
    """Start the vLLM Semantic Router server."""
    start_server(
        config=config,
        image=image,
        ...,
        readonly_dashboard=readonly_dashboard,
    )
```

#### 3.1.2 src/vllm-sr/cli/core.py

```python
def start_server(..., readonly_dashboard: bool = False):
    """Start the server with given configuration."""
    env_vars = {
        "CONFIG_PATH": config_path,
        "DASHBOARD_READONLY": "true" if readonly_dashboard else "false",
        # ... other env vars
    }
    
    # Pass to Docker container
    docker_run(env_vars=env_vars, ...)
```

---

### 3.2 启动脚本 (Shell)

#### 3.2.1 src/vllm-sr/start-dashboard.sh

```bash
#!/bin/bash

# Parse readonly mode from environment variable
READONLY_ARG=""
if [ "${DASHBOARD_READONLY}" = "true" ]; then
    READONLY_ARG="-readonly"
    echo "Starting dashboard in read-only mode..."
fi

# Start dashboard backend
exec /usr/local/bin/dashboard-backend \
    -port=8700 \
    -static=/app/frontend \
    -config=/app/config.yaml \
    -router_api=http://localhost:8080 \
    -router_metrics=http://localhost:9190/metrics \
    -envoy="http://localhost:${ENVOY_PORT}" \
    ${READONLY_ARG}
```

---

### 3.3 Go Backend

#### 3.3.1 dashboard/backend/config/config.go

```go
// Config holds all application configuration
type Config struct {
	Port          string
	StaticDir     string
	ConfigFile    string
	AbsConfigPath string
	ConfigDir     string

	// Upstream targets
	GrafanaURL    string
	PrometheusURL string
	RouterAPIURL  string
	RouterMetrics string
	JaegerURL     string
	EnvoyURL      string

	// Read-only mode (新增)
	ReadonlyMode  bool
}

// LoadConfig loads configuration from flags and environment variables
func LoadConfig() (*Config, error) {
	cfg := &Config{}

	// ... existing flags ...

	// 新增: 只读模式 flag
	readonlyMode := flag.Bool("readonly", env("DASHBOARD_READONLY", "false") == "true", "enable read-only mode")

	flag.Parse()

	// ... existing assignments ...
	cfg.ReadonlyMode = *readonlyMode  // 新增

	return cfg, nil
}
```

#### 3.3.2 dashboard/backend/handlers/config.go (修改 UpdateConfigHandler)

```go
// UpdateConfigHandler handles config updates
func UpdateConfigHandler(configPath string, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// 新增: 只读模式检查
		if readonlyMode {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			json.NewEncoder(w).Encode(map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
			})
			return
		}

		// ... 原有更新逻辑 ...
	}
}
```

#### 3.3.3 dashboard/backend/handlers/settings.go (新建)

```go
package handlers

import (
	"encoding/json"
	"net/http"

	"dashboard/backend/config"
)

// SettingsResponse represents the settings API response
type SettingsResponse struct {
	ReadonlyMode bool `json:"readonlyMode"`
}

// SettingsHandler returns dashboard settings for frontend
func SettingsHandler(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		response := SettingsResponse{
			ReadonlyMode: cfg.ReadonlyMode,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}
```

#### 3.3.4 dashboard/backend/main.go (修改路由注册)

```go
func setupRoutes(cfg *config.Config) {
	// 新增: Settings API
	http.HandleFunc("/api/settings", handlers.SettingsHandler(cfg))

	// 修改: 传递 ReadonlyMode 给 config handler
	http.HandleFunc("/api/router/config/update", handlers.UpdateConfigHandler(cfg.AbsConfigPath, cfg.ReadonlyMode))

	// ... 其他路由保持不变
}
```

---

### 3.4 需要保护的 API 端点

| 端点 | 方法 | 说明 | 只读模式行为 |
|------|------|------|-------------|
| `/api/settings` | GET | 获取 Dashboard 设置 | ✅ 允许 |
| `/api/router/config/all` | GET | 获取配置 | ✅ 允许 |
| `/api/router/config/update` | POST | 更新配置 | ❌ 403 Forbidden |
| `/api/router/v1/chat/completions` | POST | Playground 对话 | ✅ 允许 (核心功能) |
| `/api/tools-db` | GET | 获取 Tools DB | ✅ 允许 |
| `/embedded/*` | GET | Grafana/Prometheus 嵌入 | ✅ 允许 |

---

### 3.5 Frontend (React + TypeScript)

#### 3.5.1 dashboard/frontend/src/contexts/ReadonlyContext.tsx (新建)

```tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface ReadonlyContextType {
  isReadonly: boolean;
  isLoading: boolean;
}

const ReadonlyContext = createContext<ReadonlyContextType>({
  isReadonly: false,
  isLoading: true,
});

export function useReadonly(): ReadonlyContextType {
  return useContext(ReadonlyContext);
}

interface ReadonlyProviderProps {
  children: ReactNode;
}

export function ReadonlyProvider({ children }: ReadonlyProviderProps) {
  const [isReadonly, setIsReadonly] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetch('/api/settings')
      .then(res => res.json())
      .then(data => {
        setIsReadonly(data.readonlyMode || false);
        setIsLoading(false);
      })
      .catch(() => {
        setIsLoading(false);
      });
  }, []);

  return (
    <ReadonlyContext.Provider value={{ isReadonly, isLoading }}>
      {children}
    </ReadonlyContext.Provider>
  );
}
```

#### 3.5.2 dashboard/frontend/src/components/ReadonlyBanner.tsx (新建)

```tsx
import React from 'react';
import { useReadonly } from '../contexts/ReadonlyContext';
import styles from './ReadonlyBanner.module.css';

export function ReadonlyBanner() {
  const { isReadonly } = useReadonly();

  if (!isReadonly) {
    return null;
  }

  return (
    <div className={styles.banner}>
      <span className={styles.icon}>🔒</span>
      <span className={styles.text}>
        Dashboard is in read-only mode. Configuration editing is disabled.
      </span>
    </div>
  );
}
```

#### 3.5.3 dashboard/frontend/src/components/ReadonlyBanner.module.css (新建)

```css
.banner {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px 16px;
  background-color: #fff3cd;
  border-bottom: 1px solid #ffc107;
  color: #856404;
  font-size: 14px;
}

.icon {
  margin-right: 8px;
}

.text {
  font-weight: 500;
}

/* Dark mode support */
:global(.dark) .banner {
  background-color: #332701;
  border-color: #665200;
  color: #ffc107;
}
```

#### 3.5.4 dashboard/frontend/src/pages/ConfigPage.tsx (修改)

```tsx
import React from 'react';
import { useReadonly } from '../contexts/ReadonlyContext';
import { ReadonlyBanner } from '../components/ReadonlyBanner';

export function ConfigPage() {
  const { isReadonly } = useReadonly();

  const handleSave = async () => {
    if (isReadonly) {
      // Should not reach here if UI is correct
      alert('Dashboard is in read-only mode');
      return;
    }
    // ... save logic
  };

  return (
    <div>
      <ReadonlyBanner />

      {/* Config display */}
      <div className="config-content">
        {/* ... existing content ... */}
      </div>

      {/* Conditionally render edit buttons */}
      {!isReadonly && (
        <div className="actions">
          <button onClick={() => setEditModalOpen(true)}>Edit</button>
          <button onClick={() => setAddModalOpen(true)}>Add</button>
        </div>
      )}
    </div>
  );
}
```

#### 3.5.5 dashboard/frontend/src/App.tsx (修改)

```tsx
import { ReadonlyProvider } from './contexts/ReadonlyContext';

function App() {
  return (
    <ReadonlyProvider>
      <Router>
        {/* ... routes ... */}
      </Router>
    </ReadonlyProvider>
  );
}
```

---

### 3.6 错误响应格式

#### 3.6.1 只读模式 403 响应

```json
{
  "error": "readonly_mode",
  "message": "Dashboard is in read-only mode. Configuration editing is disabled."
}
```

#### 3.6.2 前端错误处理

```tsx
const handleApiError = async (response: Response) => {
  if (response.status === 403) {
    const data = await response.json();
    if (data.error === 'readonly_mode') {
      // Show readonly mode notification
      notification.warning({
        message: 'Read-only Mode',
        description: data.message,
      });
      return;
    }
  }
  // Handle other errors...
};
```

---

### 3.7 文档更新

#### 3.7.1 dashboard/README.md (补充 Security & access control 章节)

在 "Security & access control" 章节中添加：

```markdown
### Read-only Mode

For public beta deployments, you can start the dashboard in read-only mode to prevent configuration changes while still allowing access to the Playground:

```bash
# Via CLI flag
vllm-sr serve --config config.yaml --readonly-dashboard

# Via environment variable
DASHBOARD_READONLY=true ./start-dashboard.sh
```

In read-only mode:
- ✅ Playground chat is fully functional
- ✅ All configuration pages are viewable
- ❌ Edit/Add buttons are hidden
- ❌ POST to `/api/router/config/update` returns 403 Forbidden
- A banner is displayed indicating read-only mode
```

---

## 4. 测试计划

### 4.1 单元测试

| 组件 | 测试用例 |
|------|----------|
| `config.go` | 验证 `-readonly` flag 正确解析 |
| `config.go` | 验证环境变量 `DASHBOARD_READONLY` 正确读取 |
| `UpdateConfigHandler` | GET 请求正常通过 |
| `UpdateConfigHandler` | POST 请求在只读模式下返回 403 |
| `SettingsHandler` | 返回正确的 `readonlyMode` 值 |

### 4.2 集成测试

| 场景 | 操作 | 预期结果 |
|------|------|----------|
| 正常模式启动 | `vllm-sr serve` | Dashboard 完全可编辑 |
| 只读模式启动 | `vllm-sr serve --readonly-dashboard` | 编辑功能禁用 |
| 只读模式 - 查看配置 | 访问配置页面 | 正常显示，无编辑按钮 |
| 只读模式 - Playground | 发送聊天消息 | 正常工作 |
| 只读模式 - API 保护 | `POST /api/router/config/update` | 返回 403 |
| 只读模式 - Banner | 访问任意页面 | 顶部显示只读提示 |

### 4.3 E2E 测试

```bash
# 1. 启动只读模式
vllm-sr serve --config test-config.yaml --readonly-dashboard

# 2. 验证 settings API
curl http://localhost:8700/api/settings
# Expected: {"readonlyMode": true}

# 3. 验证配置 API 保护
curl -X POST http://localhost:8700/api/router/config/update -d '{"key": "value"}'
# Expected: 403 Forbidden

# 4. 验证 Playground 可用
curl -X POST http://localhost:8700/api/router/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
# Expected: 200 OK
```

---

## 5. 实现计划

### Phase 1: Backend (Go) - 预计 1 天

- [ ] `dashboard/backend/config/config.go`: 添加 `ReadonlyMode` 字段和 flag 解析
- [ ] `dashboard/backend/handlers/settings.go`: 新建 `/api/settings` 端点
- [ ] `dashboard/backend/handlers/config.go`: 添加只读模式检查
- [ ] `dashboard/backend/main.go`: 注册 settings 路由

### Phase 2: 启动链路 (Shell + Python) - 预计 0.5 天

- [ ] `src/vllm-sr/start-dashboard.sh`: 解析 `DASHBOARD_READONLY` 环境变量
- [ ] `src/vllm-sr/cli/main.py`: 添加 `--readonly-dashboard` 参数
- [ ] `src/vllm-sr/cli/core.py`: 传递环境变量到 Docker

### Phase 3: Frontend (React) - 预计 1.5 天

- [ ] `dashboard/frontend/src/contexts/ReadonlyContext.tsx`: 创建全局只读状态
- [ ] `dashboard/frontend/src/components/ReadonlyBanner.tsx`: 只读提示组件
- [ ] `dashboard/frontend/src/components/ReadonlyBanner.module.css`: Banner 样式
- [ ] `dashboard/frontend/src/pages/ConfigPage.tsx`: 条件渲染编辑功能
- [ ] `dashboard/frontend/src/App.tsx`: 添加 ReadonlyProvider

### Phase 4: 测试和文档 - 预计 0.5 天

- [ ] 单元测试
- [ ] 集成测试
- [ ] 更新 `dashboard/README.md` 文档

**总计**: 约 3.5 天

---

## 6. 后续扩展 (Out of Scope)

以下功能不在本次实现范围内，可作为后续迭代：

1. **角色权限**: 基于用户角色的细粒度权限控制 (RBAC)
2. **部分只读**: 允许某些配置可编辑，其他只读
3. **审计日志**: 记录只读模式下的访问日志
4. **密码保护**: 管理员可输入密码解锁编辑功能
5. **配置文件方式**: 通过 config.yaml 控制只读模式

---

## 7. 参考文档

- Issue: [#1110 - Add --readonly-dashboard flag](https://github.com/vllm-project/semantic-router/issues/1110)
- Dashboard Backend: `dashboard/backend/`
- Dashboard Frontend: `dashboard/frontend/`
- CLI 实现: `src/vllm-sr/cli/`
- 启动脚本: `src/vllm-sr/start-dashboard.sh`
- Dashboard README: `dashboard/README.md`
