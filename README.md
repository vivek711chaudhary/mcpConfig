# MCP Weather API Integration

This project implements a FastMCP server that integrates various external APIs, with a focus on weather data from WeatherAPI.com. It provides a collection of tools for accessing weather information, social media data, and running code execution services.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- uv package manager (recommended)
- Windows operating system

### Installation

1. Create a virtual environment and activate it:
```bash
uv venv
.venv\Scripts\activate
```

2. Install dependencies:
```bash
uv add "mcp[cli]"
```

Or using the requirements file:
```bash
uv add -r requirements.txt
```

## Features

The server implements the following features and tools:

### Weather Tools
- `get_current_weather(q)`: Get current weather for a location
- `get_weather_forecast(q, days)`: Get weather forecast for a location
- `get_weather_alerts(area)`: Get weather alerts for a location
- `search_locations(q)`: Search for locations using Weather API
- `get_time_zone(q)`: Get timezone information for a location

### Social Media Integration
- `twitter_search(query, max_results)`: Search Twitter using MCP-2 endpoint
- `enhance_tweets_masa(query, max_results, enhance_top_x, custom_instruction)`: Enhance Twitter search results using Masa API
- `enhance_tweets_playwright(tweets, custom_instruction)`: Enhance tweets using Playwright
- `enhance_tweets_playright(query, max_results, enhance_top_x, custom_instruction)`: Enhance Twitter search results using Playright API
- `extract_search_terms(query, max_results, enhance_top_x, custom_instruction)`: Extract search terms from Twitter content

### Code Execution
- `execute_python_code(code)`: Execute Python code in secure environment
- `generate_and_execute(query)`: Generate and execute Python code from natural language query

### Data Storage and Analytics
- `store_engagement_data(topic, platform, value, source, additional_metadata)`: Store engagement data for a topic
- `store_platform_engagements(topic, platform_engagements)`: Store engagement metrics from multiple platforms
- `generate_engagement_forecast(topic, platform, periods, include_history)`: Generate engagement forecast for a topic
- `get_topic_history(topic, platform)`: Get historical engagement data for a topic
- `get_topic_latest(topic)`: Get latest information for a topic

### Endpoints
- `https://playwright-mcp-620401541065.us-central1.run.app/api/topic/:topic/latest`: Get latest information for any topic (This endpoint was  forgotten to be added to the in the mcp tools. It is working fine)

### System Health Checks
- `check_mcp2_health()`: Check health status of MCP-2 Server
- `check_playwright_health()`: Check health status of Playwright MCP
- `check_prophet_health()`: Check health status of Prophet Service
- `check_code_executor_health()`: Check health status of Code Executor
- `debug_last_request()`: Debug the last API request that failed

### Resources
- `echo://{message}`: Echo a message as a resource

## Running the Server

### For Development with MCP Inspector

```bash
uv run mcp dev copi/mcpConfig/server/weather.py
```

### Running the Server Normally

```bash
uv run mcp run
```

### Installing the Server in Claude Desktop App

```bash
uv run mcp install copi/mcpConfig/server/weather.py
```

## MCP Connect in VS Code

1. Open the project folder in VS Code
2. Open terminal and run:
```bash
uv run copi/mcpConfig/server/weather.py
```
3. Press Ctrl+Shift+I to launch chat in VS Code
4. Login with GitHub and complete setup

### Adding MCP Configuration for VS Code User Settings

There are two ways to add MCP configuration for VS Code user settings:

1. **Method 1**: Through VS Code settings UI
   - Go to File > Preferences > Settings
   - Search for "mcp"
   - Add the configuration values

2. **Method 2**: Directly edit settings.json
   - Press Ctrl+Shift+P
   - Type "Preferences: Open User Settings (JSON)"
   - Add the MCP configuration to the JSON file

## API Keys and Configuration

The server uses the following external APIs:

- WeatherAPI.com: API key is required and included in the configuration
- MCP-2 Server: Used for Twitter search and enhancement
- Playwright MCP: Used for web automation and content enhancement
- Prophet Service: Used for engagement data storage and forecasting
- Code Executor: Used for secure code execution

## Error Handling

The server implements robust error handling with:

- Automatic retries with exponential backoff
- Different timeout settings for slow endpoints
- Progress notifications for long-running operations
- Detailed error messages and debugging information

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to improve the functionality.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 