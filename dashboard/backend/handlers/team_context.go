package handlers

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// --- Team Context for SOUL.md ---

// TeamContext holds all team-related information for an agent
type TeamContext struct {
	TeamName      string       // Team display name
	TeamID        string       // Team unique identifier
	SelfName      string       // This agent's display name
	SelfMention   string       // This agent's @mention handle
	SelfRole      string       // This agent's role description
	SelfKind      string       // "leader" or "worker"
	LeaderName    string       // Leader's display name (empty if no leader)
	LeaderMention string       // Leader's @mention handle (empty if no leader)
	Members       []TeamMember // All team members including self
}

// TeamMember represents a single team member
type TeamMember struct {
	Name    string // Display name
	Mention string // @mention handle (container name)
	Role    string // Role description
	Kind    string // "leader" or "worker"
	IsSelf  bool   // True if this is the current agent
}

// BuildTeamContext constructs a TeamContext from team data
func BuildTeamContext(team TeamEntry, members []ContainerEntry, selfName string) TeamContext {
	ctx := TeamContext{
		TeamName: strings.TrimSpace(team.Name),
		TeamID:   team.ID,
	}

	// Sort members by name for consistent output
	sorted := make([]ContainerEntry, len(members))
	copy(sorted, members)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })

	// Find leader
	var leader *ContainerEntry
	for i := range sorted {
		if sorted[i].TeamID == team.ID && normalizeRoleKind(sorted[i].RoleKind) == "leader" {
			leader = &sorted[i]
			break
		}
	}
	// Fallback: use team.LeaderID
	if leader == nil && team.LeaderID != "" {
		for i := range sorted {
			if sorted[i].Name == team.LeaderID {
				leader = &sorted[i]
				break
			}
		}
	}

	if leader != nil {
		ctx.LeaderName = workerDisplayName(*leader)
		ctx.LeaderMention = "@" + leader.Name
	}

	// Build member list
	for _, m := range sorted {
		if m.TeamID != team.ID {
			continue
		}
		kind := normalizeRoleKind(m.RoleKind)
		if kind != "leader" {
			kind = "worker"
		}
		role := strings.TrimSpace(m.AgentRole)
		if role == "" {
			role = kind
		}
		displayName := workerDisplayName(m)
		isSelf := m.Name == selfName
		member := TeamMember{
			Name:    displayName,
			Mention: "@" + m.Name,
			Role:    role,
			Kind:    kind,
			IsSelf:  isSelf,
		}
		ctx.Members = append(ctx.Members, member)
		if isSelf {
			ctx.SelfName = displayName
			ctx.SelfMention = "@" + m.Name
			ctx.SelfRole = role
			ctx.SelfKind = kind
		}
	}

	return ctx
}

// GenerateTeamBlock generates the Markdown team block for SOUL.md
func GenerateTeamBlock(ctx TeamContext) string {
	if ctx.TeamID == "" || len(ctx.Members) == 0 {
		return ""
	}

	var sb strings.Builder

	// Team section
	sb.WriteString("## Team\n\n")
	sb.WriteString("| Key | Value |\n")
	sb.WriteString("|-----|-------|\n")
	sb.WriteString(fmt.Sprintf("| Name | %s |\n", ctx.TeamName))
	sb.WriteString(fmt.Sprintf("| ID | %s |\n", ctx.TeamID))
	sb.WriteString("\n")

	// Role section (self info)
	sb.WriteString("## Role\n\n")
	sb.WriteString("| Key | Value |\n")
	sb.WriteString("|-----|-------|\n")
	sb.WriteString(fmt.Sprintf("| Position | %s |\n", ctx.SelfRole))
	sb.WriteString(fmt.Sprintf("| Kind | %s |\n", ctx.SelfKind))
	if ctx.LeaderMention != "" && ctx.SelfKind != "leader" {
		sb.WriteString(fmt.Sprintf("| Leader | %s |\n", ctx.LeaderMention))
	}
	sb.WriteString("\n")

	// Members section
	sb.WriteString("## Members\n\n")
	sb.WriteString("| Name | @Mention | Role | Kind |\n")
	sb.WriteString("|------|----------|------|------|\n")
	for _, m := range ctx.Members {
		name := m.Name
		if m.IsSelf {
			name = "**" + name + "**"
		}
		sb.WriteString(fmt.Sprintf("| %s | %s | %s | %s |\n", name, m.Mention, m.Role, m.Kind))
	}
	sb.WriteString("\n")

	// Rules section
	sb.WriteString("## Rules\n\n")
	if ctx.SelfKind == "leader" {
		sb.WriteString("- You are the **leader** of this team\n")
		sb.WriteString("- You can delegate tasks to team members using @mentions\n")
		sb.WriteString("- Only delegate when the user provides an explicit executable task\n")
		sb.WriteString("- Keep the team aligned and coordinate effectively\n")
	} else {
		sb.WriteString(fmt.Sprintf("- You are a **%s** in this team\n", ctx.SelfKind))
		if ctx.LeaderMention != "" {
			sb.WriteString(fmt.Sprintf("- Your leader is %s - coordinate through them\n", ctx.LeaderMention))
		}
		sb.WriteString("- Workers cannot use @mentions to delegate tasks\n")
		sb.WriteString("- Report progress in plain text without @mentions\n")
	}

	return sb.String()
}

// teamBlockMarkers defines the HTML comment markers for the team block
const (
	teamBlockBegin = "<!-- TEAM:BEGIN -->"
	teamBlockEnd   = "<!-- TEAM:END -->"
)

// teamBlockRegex matches the entire team block including markers
var teamBlockRegex = regexp.MustCompile(`(?s)` + regexp.QuoteMeta(teamBlockBegin) + `.*?` + regexp.QuoteMeta(teamBlockEnd) + `\n*`)

// UpdateSoulTeamContext updates or inserts the team block in SOUL.md
func UpdateSoulTeamContext(soulPath string, ctx TeamContext) error {
	content, err := os.ReadFile(soulPath)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to read SOUL.md: %w", err)
	}

	teamBlock := ""
	if ctx.TeamID != "" && len(ctx.Members) > 0 {
		teamBlock = teamBlockBegin + "\n" + GenerateTeamBlock(ctx) + teamBlockEnd + "\n"
	}

	existingContent := string(content)
	var newContent string

	if teamBlockRegex.MatchString(existingContent) {
		// Replace existing team block
		if teamBlock == "" {
			// Remove the block entirely
			newContent = teamBlockRegex.ReplaceAllString(existingContent, "")
		} else {
			newContent = teamBlockRegex.ReplaceAllString(existingContent, teamBlock)
		}
	} else if teamBlock != "" {
		// Append team block at the end
		newContent = strings.TrimRight(existingContent, "\n")
		if newContent != "" {
			newContent += "\n\n"
		}
		newContent += teamBlock
	} else {
		// No team block to add and none exists
		newContent = existingContent
	}

	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(soulPath), 0o755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	return os.WriteFile(soulPath, []byte(newContent), 0o644)
}

// RemoveSoulTeamContext removes the team block from SOUL.md
func RemoveSoulTeamContext(soulPath string) error {
	return UpdateSoulTeamContext(soulPath, TeamContext{})
}

// GetAgentSoulPath returns the path to SOUL.md for a given agent
func (h *OpenClawHandler) GetAgentSoulPath(containerName string) string {
	dataDir := h.containerDataDir(containerName)
	return filepath.Join(dataDir, "workspace", "SOUL.md")
}

// UpdateAgentTeamContext updates the team context in an agent's SOUL.md
func (h *OpenClawHandler) UpdateAgentTeamContext(containerName string, team TeamEntry, members []ContainerEntry) error {
	soulPath := h.GetAgentSoulPath(containerName)
	ctx := BuildTeamContext(team, members, containerName)
	return UpdateSoulTeamContext(soulPath, ctx)
}

// RemoveAgentTeamContext removes the team context from an agent's SOUL.md
func (h *OpenClawHandler) RemoveAgentTeamContext(containerName string) error {
	soulPath := h.GetAgentSoulPath(containerName)
	return RemoveSoulTeamContext(soulPath)
}

// SyncTeamMembersContext updates SOUL.md for all members of a team
func (h *OpenClawHandler) SyncTeamMembersContext(team TeamEntry, members []ContainerEntry) error {
	var lastErr error
	for _, m := range members {
		if m.TeamID != team.ID {
			continue
		}
		if err := h.UpdateAgentTeamContext(m.Name, team, members); err != nil {
			lastErr = err
		}
	}
	return lastErr
}
