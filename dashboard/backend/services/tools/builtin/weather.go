package builtin

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// Weather is a built-in tool for weather information
type Weather struct {
	client *http.Client
}

// NewWeather creates a new Weather tool
func NewWeather() *Weather {
	return &Weather{
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Name returns the tool name (matches tools_db.json)
func (w *Weather) Name() string {
	return "get_weather"
}

// Description returns the tool description
func (w *Weather) Description() string {
	return "Get current weather information for a specific location. " +
		"Returns temperature, conditions, humidity, and wind speed."
}

// Parameters returns the tool parameters
func (w *Weather) Parameters() []models.ToolParameter {
	return []models.ToolParameter{
		{
			Name:        "location",
			Type:        "string",
			Description: "City name or location, e.g., 'Beijing', 'New York, US', 'London, UK'",
			Required:    true,
		},
		{
			Name:        "unit",
			Type:        "string",
			Description: "Temperature unit: 'celsius' or 'fahrenheit'",
			Required:    false,
			Default:     "celsius",
			Enum:        []any{"celsius", "fahrenheit"},
		},
	}
}

// WeatherData represents weather information
type WeatherData struct {
	Location    string  `json:"location"`
	Temperature float64 `json:"temperature"`
	Unit        string  `json:"unit"`
	Condition   string  `json:"condition"`
	Humidity    int     `json:"humidity"`
	WindSpeed   float64 `json:"wind_speed"`
	WindUnit    string  `json:"wind_unit"`
	FeelsLike   float64 `json:"feels_like,omitempty"`
	Description string  `json:"description,omitempty"`
}

// Execute gets weather information for a location
func (w *Weather) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	location, ok := args["location"].(string)
	if !ok || location == "" {
		return nil, fmt.Errorf("location must be a non-empty string")
	}

	unit := "celsius"
	if u, ok := args["unit"].(string); ok && (u == "celsius" || u == "fahrenheit") {
		unit = u
	}

	// Check for OpenWeatherMap API key
	apiKey := os.Getenv("OPENWEATHERMAP_API_KEY")
	if apiKey != "" {
		return w.getWeatherFromAPI(ctx, location, unit, apiKey)
	}

	// Return mock weather data if no API key is configured
	return w.mockWeather(location, unit)
}

// getWeatherFromAPI fetches real weather data from OpenWeatherMap
func (w *Weather) getWeatherFromAPI(ctx context.Context, location, unit, apiKey string) (interface{}, error) {
	baseURL := "https://api.openweathermap.org/data/2.5/weather"
	params := url.Values{}
	params.Set("q", location)
	params.Set("appid", apiKey)
	if unit == "celsius" {
		params.Set("units", "metric")
	} else {
		params.Set("units", "imperial")
	}

	req, err := http.NewRequestWithContext(ctx, "GET", baseURL+"?"+params.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("weather request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("weather API error: %s", string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v", err)
	}

	// Extract weather data
	data := WeatherData{
		Location: location,
		Unit:     unit,
		WindUnit: "m/s",
	}

	if unit == "fahrenheit" {
		data.WindUnit = "mph"
	}

	if main, ok := result["main"].(map[string]interface{}); ok {
		if temp, ok := main["temp"].(float64); ok {
			data.Temperature = temp
		}
		if humidity, ok := main["humidity"].(float64); ok {
			data.Humidity = int(humidity)
		}
		if feelsLike, ok := main["feels_like"].(float64); ok {
			data.FeelsLike = feelsLike
		}
	}

	if weather, ok := result["weather"].([]interface{}); ok && len(weather) > 0 {
		if w, ok := weather[0].(map[string]interface{}); ok {
			if main, ok := w["main"].(string); ok {
				data.Condition = main
			}
			if desc, ok := w["description"].(string); ok {
				data.Description = desc
			}
		}
	}

	if wind, ok := result["wind"].(map[string]interface{}); ok {
		if speed, ok := wind["speed"].(float64); ok {
			data.WindSpeed = speed
		}
	}

	return map[string]interface{}{
		"weather": data,
		"source":  "openweathermap",
	}, nil
}

// mockWeather returns mock weather data when no API key is configured
func (w *Weather) mockWeather(location, unit string) (interface{}, error) {
	// Generate mock weather based on location
	temp := 22.0
	if unit == "fahrenheit" {
		temp = 72.0
	}

	data := WeatherData{
		Location:    location,
		Temperature: temp,
		Unit:        unit,
		Condition:   "Partly Cloudy",
		Humidity:    55,
		WindSpeed:   12.5,
		WindUnit:    "km/h",
		FeelsLike:   temp - 1,
		Description: "Partly cloudy with mild temperatures",
	}

	if unit == "fahrenheit" {
		data.WindUnit = "mph"
		data.WindSpeed = 7.8
	}

	return map[string]interface{}{
		"weather": data,
		"source":  "mock",
		"note":    "Configure OPENWEATHERMAP_API_KEY environment variable for real weather data",
	}, nil
}
