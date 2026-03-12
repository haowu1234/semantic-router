package handlers

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildTeamContext(t *testing.T) {
	team := TeamEntry{
		ID:       "physics-lab",
		Name:     "Physics Lab",
		LeaderID: "alex",
	}

	members := []ContainerEntry{
		{Name: "alex", TeamID: "physics-lab", AgentName: "Alex", AgentRole: "Team Lead", RoleKind: "leader"},
		{Name: "quark", TeamID: "physics-lab", AgentName: "Quark", AgentRole: "实验策划", RoleKind: "worker"},
		{Name: "photon", TeamID: "physics-lab", AgentName: "Photon", AgentRole: "数据分析", RoleKind: "worker"},
		{Name: "other", TeamID: "other-team", AgentName: "Other", AgentRole: "Other Role", RoleKind: "worker"},
	}

	ctx := BuildTeamContext(team, members, "quark")

	if ctx.TeamName != "Physics Lab" {
		t.Errorf("expected TeamName 'Physics Lab', got %q", ctx.TeamName)
	}
	if ctx.TeamID != "physics-lab" {
		t.Errorf("expected TeamID 'physics-lab', got %q", ctx.TeamID)
	}
	if ctx.SelfName != "Quark" {
		t.Errorf("expected SelfName 'Quark', got %q", ctx.SelfName)
	}
	if ctx.SelfMention != "@quark" {
		t.Errorf("expected SelfMention '@quark', got %q", ctx.SelfMention)
	}
	if ctx.SelfKind != "worker" {
		t.Errorf("expected SelfKind 'worker', got %q", ctx.SelfKind)
	}
	if ctx.LeaderName != "Alex" {
		t.Errorf("expected LeaderName 'Alex', got %q", ctx.LeaderName)
	}
	if ctx.LeaderMention != "@alex" {
		t.Errorf("expected LeaderMention '@alex', got %q", ctx.LeaderMention)
	}
	// Should only include members from physics-lab team
	if len(ctx.Members) != 3 {
		t.Errorf("expected 3 members, got %d", len(ctx.Members))
	}
}

func TestBuildTeamContextAsLeader(t *testing.T) {
	team := TeamEntry{
		ID:       "physics-lab",
		Name:     "Physics Lab",
		LeaderID: "alex",
	}

	members := []ContainerEntry{
		{Name: "alex", TeamID: "physics-lab", AgentName: "Alex", AgentRole: "Team Lead", RoleKind: "leader"},
		{Name: "quark", TeamID: "physics-lab", AgentName: "Quark", AgentRole: "实验策划", RoleKind: "worker"},
	}

	ctx := BuildTeamContext(team, members, "alex")

	if ctx.SelfKind != "leader" {
		t.Errorf("expected SelfKind 'leader', got %q", ctx.SelfKind)
	}
	if ctx.SelfName != "Alex" {
		t.Errorf("expected SelfName 'Alex', got %q", ctx.SelfName)
	}
}

func TestGenerateTeamBlock(t *testing.T) {
	ctx := TeamContext{
		TeamName:      "Physics Lab",
		TeamID:        "physics-lab",
		SelfName:      "Quark",
		SelfMention:   "@quark",
		SelfRole:      "实验策划",
		SelfKind:      "worker",
		LeaderName:    "Alex",
		LeaderMention: "@alex",
		Members: []TeamMember{
			{Name: "Alex", Mention: "@alex", Role: "Team Lead", Kind: "leader", IsSelf: false},
			{Name: "Quark", Mention: "@quark", Role: "实验策划", Kind: "worker", IsSelf: true},
		},
	}

	block := GenerateTeamBlock(ctx)

	// Check essential parts
	if !strings.Contains(block, "## Team") {
		t.Error("missing '## Team' section")
	}
	if !strings.Contains(block, "| Physics Lab |") {
		t.Error("missing team name in table")
	}
	if !strings.Contains(block, "## Role") {
		t.Error("missing '## Role' section")
	}
	if !strings.Contains(block, "| worker |") {
		t.Error("missing worker kind in role table")
	}
	if !strings.Contains(block, "| @alex |") {
		t.Error("missing leader mention in role table")
	}
	if !strings.Contains(block, "## Members") {
		t.Error("missing '## Members' section")
	}
	if !strings.Contains(block, "**Quark**") {
		t.Error("missing bold self name in members table")
	}
	if !strings.Contains(block, "## Rules") {
		t.Error("missing '## Rules' section")
	}
	if !strings.Contains(block, "Workers cannot use @mentions") {
		t.Error("missing worker rule")
	}
}

func TestGenerateTeamBlockAsLeader(t *testing.T) {
	ctx := TeamContext{
		TeamName:      "Physics Lab",
		TeamID:        "physics-lab",
		SelfName:      "Alex",
		SelfMention:   "@alex",
		SelfRole:      "Team Lead",
		SelfKind:      "leader",
		LeaderName:    "Alex",
		LeaderMention: "@alex",
		Members: []TeamMember{
			{Name: "Alex", Mention: "@alex", Role: "Team Lead", Kind: "leader", IsSelf: true},
			{Name: "Quark", Mention: "@quark", Role: "实验策划", Kind: "worker", IsSelf: false},
		},
	}

	block := GenerateTeamBlock(ctx)

	if !strings.Contains(block, "You are the **leader**") {
		t.Error("missing leader rule")
	}
	if !strings.Contains(block, "delegate tasks") {
		t.Error("missing delegation instruction")
	}
}

func TestUpdateSoulTeamContext(t *testing.T) {
	tmpDir := t.TempDir()
	soulPath := filepath.Join(tmpDir, "SOUL.md")

	// Create initial SOUL.md
	initialContent := `# SOUL.md - Who You Are

## Core Identity

You are **Quark**, an experimental AI assistant.

## Core Truths

- Always be helpful
- Stay curious
`
	if err := os.WriteFile(soulPath, []byte(initialContent), 0o644); err != nil {
		t.Fatalf("failed to write initial SOUL.md: %v", err)
	}

	// Add team context
	ctx := TeamContext{
		TeamName:      "Physics Lab",
		TeamID:        "physics-lab",
		SelfName:      "Quark",
		SelfMention:   "@quark",
		SelfRole:      "实验策划",
		SelfKind:      "worker",
		LeaderName:    "Alex",
		LeaderMention: "@alex",
		Members: []TeamMember{
			{Name: "Alex", Mention: "@alex", Role: "Team Lead", Kind: "leader", IsSelf: false},
			{Name: "Quark", Mention: "@quark", Role: "实验策划", Kind: "worker", IsSelf: true},
		},
	}

	if err := UpdateSoulTeamContext(soulPath, ctx); err != nil {
		t.Fatalf("UpdateSoulTeamContext failed: %v", err)
	}

	// Verify content
	content, err := os.ReadFile(soulPath)
	if err != nil {
		t.Fatalf("failed to read SOUL.md: %v", err)
	}

	contentStr := string(content)
	if !strings.Contains(contentStr, "<!-- TEAM:BEGIN -->") {
		t.Error("missing TEAM:BEGIN marker")
	}
	if !strings.Contains(contentStr, "<!-- TEAM:END -->") {
		t.Error("missing TEAM:END marker")
	}
	if !strings.Contains(contentStr, "## Core Identity") {
		t.Error("original content was lost")
	}
	if !strings.Contains(contentStr, "## Team") {
		t.Error("team section not added")
	}
}

func TestUpdateSoulTeamContextReplace(t *testing.T) {
	tmpDir := t.TempDir()
	soulPath := filepath.Join(tmpDir, "SOUL.md")

	// Create SOUL.md with existing team block
	existingContent := `# SOUL.md

## Core Identity

You are **Quark**.

<!-- TEAM:BEGIN -->
## Team

| Key | Value |
|-----|-------|
| Name | Old Team |
<!-- TEAM:END -->
`
	if err := os.WriteFile(soulPath, []byte(existingContent), 0o644); err != nil {
		t.Fatalf("failed to write SOUL.md: %v", err)
	}

	// Update with new team context
	ctx := TeamContext{
		TeamName:    "New Team",
		TeamID:      "new-team",
		SelfName:    "Quark",
		SelfMention: "@quark",
		SelfRole:    "worker",
		SelfKind:    "worker",
		Members: []TeamMember{
			{Name: "Quark", Mention: "@quark", Role: "worker", Kind: "worker", IsSelf: true},
		},
	}

	if err := UpdateSoulTeamContext(soulPath, ctx); err != nil {
		t.Fatalf("UpdateSoulTeamContext failed: %v", err)
	}

	content, err := os.ReadFile(soulPath)
	if err != nil {
		t.Fatalf("failed to read SOUL.md: %v", err)
	}

	contentStr := string(content)
	if strings.Contains(contentStr, "Old Team") {
		t.Error("old team name should be replaced")
	}
	if !strings.Contains(contentStr, "New Team") {
		t.Error("new team name should be present")
	}
	// Should only have one TEAM:BEGIN marker
	if strings.Count(contentStr, "<!-- TEAM:BEGIN -->") != 1 {
		t.Error("should have exactly one TEAM:BEGIN marker")
	}
}

func TestRemoveSoulTeamContext(t *testing.T) {
	tmpDir := t.TempDir()
	soulPath := filepath.Join(tmpDir, "SOUL.md")

	// Create SOUL.md with team block
	existingContent := `# SOUL.md

## Core Identity

You are **Quark**.

<!-- TEAM:BEGIN -->
## Team

| Key | Value |
|-----|-------|
| Name | Physics Lab |
<!-- TEAM:END -->

## Vibe

Stay curious.
`
	if err := os.WriteFile(soulPath, []byte(existingContent), 0o644); err != nil {
		t.Fatalf("failed to write SOUL.md: %v", err)
	}

	if err := RemoveSoulTeamContext(soulPath); err != nil {
		t.Fatalf("RemoveSoulTeamContext failed: %v", err)
	}

	content, err := os.ReadFile(soulPath)
	if err != nil {
		t.Fatalf("failed to read SOUL.md: %v", err)
	}

	contentStr := string(content)
	if strings.Contains(contentStr, "<!-- TEAM:BEGIN -->") {
		t.Error("TEAM:BEGIN marker should be removed")
	}
	if strings.Contains(contentStr, "<!-- TEAM:END -->") {
		t.Error("TEAM:END marker should be removed")
	}
	if !strings.Contains(contentStr, "## Core Identity") {
		t.Error("other content should be preserved")
	}
	if !strings.Contains(contentStr, "## Vibe") {
		t.Error("content after team block should be preserved")
	}
}

func TestGenerateTeamBlockEmpty(t *testing.T) {
	ctx := TeamContext{}
	block := GenerateTeamBlock(ctx)
	if block != "" {
		t.Errorf("expected empty block for empty context, got %q", block)
	}
}
