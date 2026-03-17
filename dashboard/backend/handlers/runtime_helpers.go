package handlers

import (
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/runtimecontrol"
)

func runtimeController() *runtimecontrol.Controller {
	return runtimecontrol.NewController()
}

func runtimeControllerForContainer(containerName string) *runtimecontrol.Controller {
	if strings.TrimSpace(containerName) == "" {
		return runtimeController()
	}
	return runtimecontrol.NewControllerForContainer(containerName)
}
