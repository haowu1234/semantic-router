package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// --- Skills ---

func (h *OpenClawHandler) SkillsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		skills, err := h.loadSkills()
		if err != nil {
			log.Printf("Warning: failed to load skills config: %v", err)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte("[]"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(skills); err != nil {
			log.Printf("openclaw: skills encode error: %v", err)
		}
	}
}

func (h *OpenClawHandler) loadSkills() ([]SkillTemplate, error) {
	candidates := make([]string, 0, 12)
	if p := strings.TrimSpace(os.Getenv("OPENCLAW_SKILLS_PATH")); p != "" {
		candidates = append(candidates, p)
	}

	candidates = append(candidates,
		filepath.Join(h.dataDir, "skills.json"),
		filepath.Join(h.dataDir, "..", "..", "config", "openclaw-skills.json"),
		"/app/config/openclaw-skills.json",
		"/app/dashboard/backend/config/openclaw-skills.json",
		"./config/openclaw-skills.json",
	)

	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(wd, "config", "openclaw-skills.json"))
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(exeDir, "config", "openclaw-skills.json"),
			filepath.Join(exeDir, "..", "config", "openclaw-skills.json"),
		)
	}

	seen := make(map[string]struct{}, len(candidates))
	for _, rawPath := range candidates {
		configPath := strings.TrimSpace(rawPath)
		if configPath == "" {
			continue
		}
		cleanPath := filepath.Clean(configPath)
		if _, ok := seen[cleanPath]; ok {
			continue
		}
		seen[cleanPath] = struct{}{}

		data, err := os.ReadFile(configPath)
		if err != nil {
			continue
		}
		var skills []SkillTemplate
		if err := json.Unmarshal(data, &skills); err != nil {
			return nil, fmt.Errorf("invalid %s: %w", configPath, err)
		}
		log.Printf("openclaw: loaded %d skills from %s", len(skills), configPath)
		return skills, nil
	}
	return []SkillTemplate{}, nil
}

func (h *OpenClawHandler) fetchSkillContent(skillID, baseImage string) string {
	containerPaths := []string{
		"/app/skills/" + skillID + "/SKILL.md",
		"/app/extensions/" + skillID + "/SKILL.md",
	}
	for _, p := range containerPaths {
		out, err := h.containerOutput("run", "--rm", baseImage, "cat", p)
		if err == nil && len(out) > 0 {
			return string(out)
		}
	}
	skills, err := h.loadSkills()
	if err != nil {
		return ""
	}
	for _, s := range skills {
		if s.ID == skillID {
			return fmt.Sprintf("---\nname: %s\ndescription: %q\nuser-invocable: true\n---\n\n# %s\n\n%s\n",
				s.ID, s.Description, s.Name, s.Description)
		}
	}
	return ""
}

// TeamNotifyContext holds the context needed for generating team-notify skill content
type TeamNotifyContext struct {
	TeamName     string
	TeamID       string
	RoomID       string
	RoomName     string
	GatewayURL   string
	AgentName    string
}

// generateTeamNotifySkillContent generates dynamic content for the team-notify skill
// This skill allows agents to proactively send messages to the team Matrix room
func generateTeamNotifySkillContent(ctx TeamNotifyContext) string {
	return fmt.Sprintf(`---
name: team-notify
description: "Send proactive messages to team Matrix room"
user-invocable: true
---

# Team Notification

Send messages to your team's Matrix room for notifications, reports, scheduled task results, and alerts.

## Team Information

| Field | Value |
|-------|-------|
| Team Name | %s |
| Team ID | %s |
| Room ID | %s |
| Room Name | %s |

## How to Send a Message

Use curl to send a message to the team room via the Dashboard API:

`+"`"+`bash
curl -X POST "%s/api/openclaw/rooms/%s/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "YOUR_MESSAGE_HERE",
    "senderType": "worker",
    "senderName": "%s"
  }'
`+"`"+`

## Example Usage

### Send a simple notification
`+"`"+`bash
curl -X POST "%s/api/openclaw/rooms/%s/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "✅ Daily report generated successfully", "senderType": "worker", "senderName": "%s"}'
`+"`"+`

### Send a scheduled task result
`+"`"+`bash
curl -X POST "%s/api/openclaw/rooms/%s/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "📊 Weekly metrics:\n- Tasks completed: 42\n- Success rate: 98%%", "senderType": "worker", "senderName": "%s"}'
`+"`"+`

### Send an alert
`+"`"+`bash
curl -X POST "%s/api/openclaw/rooms/%s/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "⚠️ Alert: Disk usage above 80%%", "senderType": "worker", "senderName": "%s"}'
`+"`"+`

## When to Use

- **Scheduled Tasks**: Report results of cron jobs or scheduled operations
- **Daily/Weekly Reports**: Send automated summaries to the team
- **Alerts**: Notify the team of important events or issues
- **Status Updates**: Keep the team informed of long-running task progress

## Notes

- Messages sent via this method appear in the team's Matrix room
- All team members will see the notification
- Use @mention syntax (e.g., @leader) to notify specific members
`,
		ctx.TeamName, ctx.TeamID, ctx.RoomID, ctx.RoomName,
		ctx.GatewayURL, ctx.RoomID, ctx.AgentName,
		ctx.GatewayURL, ctx.RoomID, ctx.AgentName,
		ctx.GatewayURL, ctx.RoomID, ctx.AgentName,
		ctx.GatewayURL, ctx.RoomID, ctx.AgentName,
	)
}
