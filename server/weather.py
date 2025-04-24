from typing import Any, List, Optional, Generator, Union, Dict
import httpx
import json
import asyncio
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# Initialize FastMCP server with basic configuration
mcp = FastMCP(name="mcp-integration")

# Constants with increased timeout
API_TIMEOUT = 60.0  # 60 seconds timeout for all API calls
API_SLOW_ENDPOINT_TIMEOUT = 120.0  # 2 minutes for slow endpoints
WEATHER_API_KEY = "174c5941cd1842f58d475356242605"
WEATHER_API_BASE = "http://api.weatherapi.com/v1"
NWS_API_BASE = "https://api.weather.gov"
MCP2_BASE_URL = "http://34.31.55.189:8080"
PLAYWRIGHT_MCP_URL = "https://playwright-mcp-nttc25y22a-uc.a.run.app"
PROPHET_SERVICE_URL = "http://34.45.252.228:8000"
CODE_EXECUTOR_URL = "http://34.66.53.176:8002"
USER_AGENT = "mcp-integration/1.0"

# Custom progress notification handler
async def send_progress(message: str):
    """Send progress notification in a way compatible with the current FastMCP version."""
    # Log progress to stdout for debugging
    print(f"PROGRESS: {message}")
    # In newer versions of FastMCP, we would use mcp.progress here
    # For now, we'll just log the progress

async def make_api_request(
    url: str, 
    method: str = "GET", 
    json_data: Optional[dict] = None,
    params: Optional[dict] = None,
    progress_callback: Optional[callable] = None
) -> Optional[dict]:
    """Make an API request with extended timeout and error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    
    # Determine if this is a potentially slow endpoint
    is_slow_endpoint = any(keyword in url for keyword in ["enhance", "playwright", "masa", "search", "execute"])
    current_timeout = API_SLOW_ENDPOINT_TIMEOUT if is_slow_endpoint else API_TIMEOUT
    
    # Send initial progress notification
    if progress_callback:
        await progress_callback(f"Starting request to {url}")
    
    # Implementation of exponential backoff for retries
    max_retries = 3 if is_slow_endpoint else 1
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            if retry_count > 0 and progress_callback:
                await progress_callback(f"Retry attempt {retry_count}/{max_retries} for {url}")
            
            async with httpx.AsyncClient(timeout=current_timeout) as client:
                if progress_callback:
                    await progress_callback(f"Connecting to {url}")
                    
                if method == "GET":
                    response = await client.get(url, headers=headers, params=params)
                else:
                    response = await client.post(url, headers=headers, json=json_data)
                
                if progress_callback:
                    await progress_callback(f"Received response from {url} with status {response.status_code}")
                    
                response.raise_for_status()
                return response.json()
                
        except httpx.TimeoutException as e:
            last_error = e
            print(f"Request to {url} timed out after {current_timeout}s on attempt {retry_count + 1}/{max_retries + 1}")
            
            # Increase timeout for next retry
            current_timeout *= 1.5
            
            if retry_count == max_retries:
                print(f"Maximum retries reached for {url}. Last error: {str(e)}")
                break
                
        except httpx.HTTPStatusError as e:
            last_error = e
            print(f"HTTP error for {url}: {e.response.status_code} - {e.response.text}")
            
            # Don't retry for client errors (4xx)
            if e.response.status_code >= 400 and e.response.status_code < 500:
                break
                
        except Exception as e:
            last_error = e
            print(f"Error making request to {url}: {str(e)}")
            
            # For other errors, we'll retry as well
        
        retry_count += 1
        
        if retry_count <= max_retries:
            # Wait before retrying with exponential backoff
            backoff_time = 2 ** retry_count
            if progress_callback:
                await progress_callback(f"Waiting {backoff_time}s before retry")
            await asyncio.sleep(backoff_time)
    
    if last_error:
        print(f"Final error for {url}: {str(last_error)}")
    
    return None

# Weather API specific request function with progress support
async def weather_api_request(endpoint: str, params: dict) -> Optional[dict]:
    """Make a direct API request to the WeatherAPI.com with minimal overhead and error handling."""
    url = f"{WEATHER_API_BASE}/{endpoint}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json"
    }
    
    # Add API key to params
    params["key"] = WEATHER_API_KEY
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Weather API error for {url}: {str(e)}")
        return None

# Health Check Endpoints
@mcp.tool()
async def check_mcp2_health() -> str:
    """Check health status of MCP-2 Server"""
    data = await make_api_request(f"{MCP2_BASE_URL}/health")
    if not data:
        return "Failed to get MCP-2 health status"
    
    return f"""MCP-2 Server Health:
Status: {data.get('status', 'unknown')}
Version: {data.get('version', 'unknown')}
Timestamp: {data.get('timestamp', 'unknown')}
Config: {data.get('config', {})}
"""

@mcp.tool()
async def check_playwright_health() -> str:
    """Check health status of Playwright MCP"""
    data = await make_api_request(f"{PLAYWRIGHT_MCP_URL}/health")
    if not data:
        return "Failed to get Playwright MCP health status"
    
    return f"""Playwright MCP Health:
Status: {data.get('status', 'unknown')}
Environment: {data.get('environment', 'unknown')}
Timestamp: {data.get('timestamp', 'unknown')}
"""

@mcp.tool()
async def check_prophet_health() -> str:
    """Check health status of Prophet Service"""
    data = await make_api_request(f"{PROPHET_SERVICE_URL}/")
    if not data:
        return "Failed to get Prophet Service health status"
    
    return f"""Prophet Service Health:
Status: {data.get('status', 'unknown')}
Message: {data.get('message', 'unknown')}
"""

@mcp.tool()
async def check_code_executor_health() -> str:
    """Check health status of Code Executor"""
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(f"{CODE_EXECUTOR_URL}/health")
            response.raise_for_status()
            return "Code Executor is healthy"
    except Exception as e:
        return f"Code Executor health check failed: {str(e)}"

# MCP-2 Server Endpoints
@mcp.tool()
async def twitter_search(query: str, max_results: int = 5) -> str:
    """Search Twitter using MCP-2 endpoint"""
    payload = {
        "tool": "twitter_search",
        "parameters": {
            "query": query,
            "max_results": max_results
        }
    }
    data = await make_api_request(f"{MCP2_BASE_URL}/mcp", method="POST", json_data=payload)
    
    if not data:
        return "Failed to perform Twitter search"
    
    return "\n\n".join([
        f"@{tweet.get('Author', {}).get('Username', 'unknown')}: {tweet.get('Content', '')}\n"
        f"URL: {tweet.get('URL', '')}\n"
        f"Metrics: {tweet.get('Metadata', {}).get('public_metrics', {})}"
        for tweet in data
    ])

@mcp.tool()
async def enhance_tweets_masa(
    query: str, 
    max_results: int = 10, 
    enhance_top_x: int = 5,
    custom_instruction: Optional[str] = None
) -> str:
    """Enhance Twitter search results using Masa API"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    payload = {
        "query": query,
        "max_results": max_results,
        "enhance_top_x": enhance_top_x
    }
    if custom_instruction:
        payload["custom_instruction"] = custom_instruction
    
    await report_progress(f"Starting search for '{query}'")
    
    data = await make_api_request(
        f"{MCP2_BASE_URL}/api/masa/enhance",
        method="POST",
        json_data=payload,
        progress_callback=report_progress
    )
    
    if not data or "results" not in data:
        return "Failed to enhance tweets"
    
    await report_progress(f"Processing {len(data['results'])} results")
    
    results = []
    for item in data["results"]:
        tweet = item.get("original_tweet", {})
        enhanced = item.get("enhanced_version", "No enhanced content")
        research = item.get("research", {})
        
        results.append(
            f"Original Tweet: {tweet.get('Content', '')}\n"
            f"Enhanced: {enhanced}\n"
            f"Research Query: {research.get('generated_query', '')}\n"
            f"Source URL: {research.get('source_url', '')}\n"
            "---"
        )
    
    await report_progress("Completed enhancement")
    return "\n".join(results)

@mcp.tool()
async def extract_search_terms(query: str, max_results: int = 5, enhance_top_x: int = 3, custom_instruction: Optional[str] = None) -> str:
    """Extract search terms from Twitter content"""
    payload = {
        "query": query,
        "max_results": max_results,
        "enhance_top_x": enhance_top_x
    }
    if custom_instruction:
        payload["custom_instruction"] = custom_instruction
    
    data = await make_api_request(
        f"{MCP2_BASE_URL}/api/masa/searchTerm",
        method="POST",
        json_data=payload
    )
    
    if not data or "results" not in data:
        return "Failed to extract search terms"
    
    results = []
    for item in data["results"]:
        tweet = item.get("original_tweet", {})
        search_term = item.get("search_term", "")
        
        results.append(
            f"Original: {tweet.get('Content', '')}\n"
            f"Search Term: {search_term}\n"
            "---"
        )
    
    return "\n".join(results)

# Playwright MCP Endpoints
@mcp.tool()
async def enhance_tweets_playwright(tweets: List[dict], custom_instruction: Optional[str] = None) -> str:
    """Enhance tweets using Playwright"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    payload = {
        "tweets": tweets
    }
    if custom_instruction:
        payload["custom_instruction"] = custom_instruction
    
    await report_progress(f"Starting enhancement of {len(tweets)} tweets")
    
    data = await make_api_request(
        f"{PLAYWRIGHT_MCP_URL}/enhance-tweets-playwright",
        method="POST",
        json_data=payload,
        progress_callback=report_progress
    )
    
    if not data or not data.get("success"):
        return "Failed to enhance tweets with Playwright"
    
    await report_progress(f"Processing {len(data.get('results', []))} results")
    
    results = []
    for item in data.get("results", []):
        original = item.get("original_tweet", {})
        enhanced = item.get("enhanced_version", "")
        research = item.get("research", {})
        error = item.get("error", "")
        
        if error:
            results.append(
                f"Original: {original.get('Content', '')}\n"
                f"Error: {error}\n"
                f"Details: {item.get('details', '')}\n"
                "---"
            )
        else:
            results.append(
                f"Original: {original.get('Content', '')}\n"
                f"Enhanced: {enhanced}\n"
                f"Research Query: {research.get('generated_query', '')}\n"
                f"Source URL: {research.get('source_url', '')}\n"
                "---"
            )
    
    performance = data.get("performance", {})
    await report_progress("Enhancement completed")
    return f"Success: {data.get('success')}\nCount: {data.get('count')}\nPerformance: {performance.get('avg_time_per_tweet_ms')}ms per tweet\n\n" + "\n".join(results)

@mcp.tool()
async def enhance_tweets_playright(
    query: str, 
    max_results: int = 8, 
    enhance_top_x: int = 3,
    custom_instruction: Optional[str] = None
) -> str:
    """Enhance Twitter search results using Playright API"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    payload = {
        "query": query,
        "max_results": max_results,
        "enhance_top_x": enhance_top_x
    }
    if custom_instruction:
        payload["custom_instruction"] = custom_instruction
    
    await report_progress(f"Starting search for '{query}'")
    
    data = await make_api_request(
        f"{MCP2_BASE_URL}/api/playright/enhance",
        method="POST",
        json_data=payload,
        progress_callback=report_progress
    )
    
    if not data or "results" not in data:
        await report_progress("Failed to enhance tweets with Playright")
        return "Failed to enhance tweets with Playright"
    
    await report_progress(f"Processing {len(data['results'])} results")
    
    results = []
    for item in data["results"]:
        tweet = item.get("original_tweet", {})
        enhanced = item.get("enhanced_version", "No enhanced content")
        research = item.get("research", {})
        
        results.append(
            f"Original Tweet: {tweet.get('Content', '')}\n"
            f"Enhanced: {enhanced}\n"
            f"Research Query: {research.get('generated_query', '')}\n"
            f"Source URL: {research.get('source_url', '')}\n"
            "---"
        )
    
    await report_progress("Completed enhancement")
    return "\n".join(results)

# Code Executor Endpoints
@mcp.tool()
async def execute_python_code(code: str) -> str:
    """Execute Python code in secure environment"""
    async def report_progress(message: str):
        await send_progress(message)
        
    payload = {
        "code": code,
        "timeout": 10,
        "memory_limit": "200m",
        "cpu_limit": 0.5,
        "validate_code": True
    }
    
    await report_progress("Preparing to execute Python code")
    
    data = await make_api_request(
        f"{CODE_EXECUTOR_URL}/execute",
        method="POST",
        json_data=payload,
        progress_callback=report_progress
    )
    
    if not data:
        await report_progress("Failed to execute code")
        return "Failed to execute code"
    
    await report_progress("Code execution completed")
    
    return (
        f"Code Execution Results:\n"
        f"-----------------------------------\n"
        f"Output:\n{data.get('stdout', '')}\n"
        f"Errors:\n{data.get('stderr', '')}\n"
        f"Exit Code: {data.get('exit_code', -1)}\n"
        f"Execution Time: {data.get('execution_time', 0)}s\n"
        f"Validation: {data.get('validation_result', '')}"
    )

@mcp.tool()
async def generate_and_execute(query: str) -> str:
    """Generate and execute Python code from natural language query"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    payload = {
        "query": query,
        "timeout": 10,
        "memory_limit": "300m",
        "cpu_limit": 0.5
    }
    
    await report_progress(f"Generating code for: {query}")
    
    data = await make_api_request(
        f"{CODE_EXECUTOR_URL}/generate-and-execute",
        method="POST",
        json_data=payload,
        progress_callback=report_progress
    )
    
    if not data:
        return "Failed to generate and execute code"
    
    await report_progress("Code generated and executed")
    
    return (
        f"Generated Code:\n{data.get('generated_code', '')}\n\n"
        f"Execution Results:\n"
        f"-----------------------------------\n"
        f"Output:\n{data.get('stdout', '')}\n"
        f"Errors:\n{data.get('stderr', '')}\n"
        f"Exit Code: {data.get('exit_code', -1)}\n"
        f"Execution Time: {data.get('execution_time', 0)}s\n"
        f"Validation: {data.get('validation_result', '')}"
    )

# Prophet Service Endpoints
@mcp.tool()
async def store_engagement_data(
    topic: str,
    platform: str,
    value: float,
    source: str = "api",
    additional_metadata: Optional[dict] = None
) -> str:
    """Store engagement data for a topic"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    metadata = {
        "source": source
    }
    if additional_metadata:
        metadata.update(additional_metadata)
    
    payload = {
        "topic": topic,
        "platform": platform,
        "timestamp": datetime.utcnow().isoformat(),
        "value": value,
        "metadata": metadata
    }
    
    await report_progress(f"Storing engagement data for {topic} on {platform}")
    
    data = await make_api_request(
        f"{PROPHET_SERVICE_URL}/api/v1/store-engagement",
        method="POST",
        json_data=payload,
        progress_callback=report_progress
    )
    
    if not data:
        await report_progress("Failed to store engagement data")
        return "Failed to store engagement data"
    
    await report_progress("Engagement data stored successfully")
    return f"Status: {data.get('status', 'unknown')}\nMessage: {data.get('message', '')}"

@mcp.tool()
async def store_platform_engagements(
    topic: str,
    platform_engagements: List[dict]
) -> str:
    """Store engagement metrics from multiple platforms"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    payload = {
        "topic": topic,
        "timestamp": datetime.utcnow().isoformat(),
        "results": platform_engagements,
        "stats": {
            "platform_status": {
                platform: "success" for platform in set(
                    e.get("platform", "unknown") for e in platform_engagements
                )
            }
        }
    }
    
    platforms = set(e.get("platform", "unknown") for e in platform_engagements)
    await report_progress(f"Storing engagement data for {topic} on {', '.join(platforms)}")
    
    data = await make_api_request(
        f"{PROPHET_SERVICE_URL}/api/v1/store-platform-engagements",
        method="POST",
        json_data=payload,
        progress_callback=report_progress
    )
    
    if not data:
        await report_progress("Failed to store platform engagements")
        return "Failed to store platform engagements"
    
    await report_progress("Platform engagements stored successfully")
    return (
        f"Status: {data.get('status', 'unknown')}\n"
        f"Message: {data.get('message', '')}\n"
        f"Platforms: {data.get('platforms', [])}"
    )

@mcp.tool()
async def generate_engagement_forecast(
    topic: str,
    platform: str = "twitter",
    periods: int = 7,
    include_history: bool = True
) -> str:
    """Generate engagement forecast for a topic"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    payload = {
        "topic": topic,
        "platform": platform,
        "periods": periods,
        "frequency": "D",
        "include_history": include_history
    }
    
    await report_progress(f"Generating forecast for {topic} on {platform} for {periods} days")
    
    data = await make_api_request(
        f"{PROPHET_SERVICE_URL}/api/v1/forecast",
        method="POST",
        json_data=payload,
        progress_callback=report_progress
    )
    
    if not data:
        await report_progress("Failed to generate forecast")
        return "Failed to generate forecast"
    
    await report_progress("Processing forecast data")
    
    forecast = "\n".join(
        f"{date}: {value}" 
        for date, value in zip(data.get("forecast_dates", []), data.get("forecast_values", []))
    )
    
    history = "\n".join(
        f"{date}: {value}" 
        for date, value in zip(data.get("historical_dates", []), data.get("historical_values", []))
    )
    
    await report_progress("Forecast generation completed")
    return (
        f"Forecast for {topic} on {platform}:\n{forecast}\n\n"
        f"Historical data:\n{history}"
    )

@mcp.tool()
async def get_topic_history(topic: str, platform: Optional[str] = None) -> str:
    """Get historical engagement data for a topic"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    params = {}
    if platform:
        params["platform"] = platform
    
    await report_progress(f"Fetching history for {topic}" + (f" on {platform}" if platform else ""))
    
    data = await make_api_request(
        f"{PROPHET_SERVICE_URL}/api/v1/topics/{topic}/history",
        params=params,
        progress_callback=report_progress
    )
    
    if not data:
        await report_progress(f"No historical data found for topic {topic}")
        return f"No historical data found for topic {topic}"
    
    await report_progress("Processing historical data")
    
    history = "\n".join(
        f"{item.get('timestamp', '')}: {item.get('value', 0)} "
        f"(items: {item.get('metadata', {}).get('items_count', 0)})"
        for item in data.get("data", [])
    )
    
    await report_progress("History retrieval completed")
    return (
        f"History for {topic} ({data.get('platform', 'all platforms')}):\n"
        f"{history}"
    )

# Original Weather Tools (kept for compatibility)
@mcp.tool()
async def get_weather_alerts(area: str) -> str:
    """Get weather alerts for a location"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    await report_progress(f"Fetching weather alerts for {area}")
    
    data = await weather_api_request("forecast.json", {"q": area, "days": 1, "alerts": "yes"})
    
    if not data or "alerts" not in data:
        await report_progress("Unable to fetch alerts")
        return "Unable to fetch alerts or no alerts found."
    
    alerts_data = data.get("alerts", {}).get("alert", [])
    
    if not alerts_data:
        await report_progress("No active alerts for this area")
        return "No active alerts for this area."
    
    await report_progress(f"Processing {len(alerts_data)} alerts")
    
    alerts = []
    for alert in alerts_data:
        alerts.append(
            f"Event: {alert.get('event', 'Unknown')}\n"
            f"Severity: {alert.get('severity', 'Unknown')}\n"
            f"Areas: {alert.get('areas', 'Unknown')}\n"
            f"Category: {alert.get('category', 'Unknown')}\n"
            f"Effective: {alert.get('effective', 'Unknown')}\n"
            f"Expires: {alert.get('expires', 'Unknown')}\n"
            f"Description: {alert.get('desc', 'No description')}\n"
            "---"
        )
    
    await report_progress("Alerts processing completed")
    return "\n".join(alerts)

@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"

@mcp.tool()
async def get_topic_latest(topic: str) -> str:
    """Get latest information for a topic"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    await report_progress(f"Fetching latest information for {topic}")
    
    data = await make_api_request(
        f"{PLAYWRIGHT_MCP_URL}/api/topic/{topic}/latest",
        progress_callback=report_progress
    )
    
    if not data:
        await report_progress(f"No latest data found for topic {topic}")
        return f"No latest data found for topic {topic}"
    
    await report_progress("Processing latest topic data")
    
    # Format the response based on what the API returns
    if isinstance(data, dict):
        result = f"Latest information for {topic}:\n{json.dumps(data, indent=2)}"
    else:
        result = f"Latest information for {topic}:\n{data}"
    
    await report_progress("Latest information retrieved")
    return result

@mcp.tool()
async def debug_last_request() -> str:
    """Debug the last API request that failed"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    await report_progress("Generating debug information for last request")
    
    debug_info = f"""
To debug API requests that are timing out (MCP error -32001):

1. Check API endpoint is correct:
   - enhace-tweets-playwright → {PLAYWRIGHT_MCP_URL}/enhance-tweets-playwright
   - masa enhance → {MCP2_BASE_URL}/api/masa/enhance
   - store engagement → {PROPHET_SERVICE_URL}/api/v1/store-engagement
   - weather API → {WEATHER_API_BASE} (with key {WEATHER_API_KEY})

2. Verify request payload format:
   - For enhance-tweets-playwright: Use "tweets" list with tweet objects
   - For masa enhance: Use "query", "max_results", "enhance_top_x"
   - For playright enhance: Use "query", "max_results", "enhance_top_x" 
   - For weather API: Make sure to include "key" parameter

3. Current timeout setting is {API_TIMEOUT} seconds
   - These APIs may need more time to process, consider increasing timeout

4. Error handling strategies:
   - Retry with fewer results (lower max_results)
   - Check server status with health endpoints
   - Break requests into smaller batches

5. API Server status:
   - MCP2 Server: Run check_mcp2_health()
   - Playwright: Run check_playwright_health()
   - Prophet Service: Run check_prophet_health()
"""
    
    await report_progress("Debug information generated")
    return debug_info

# WeatherAPI.com Integration
@mcp.tool()
async def get_current_weather(q: str) -> str:
    """
    Get current weather for a location.
    
    Parameters:
    - q: Location query (city name, lat/lon, IP address, US zip, UK postcode, etc.)
    """
    await send_progress(f"Fetching current weather for {q}")
    data = await weather_api_request("current.json", {"q": q})
    
    if not data:
        await send_progress(f"Failed to get current weather for {q}")
        return f"Failed to get current weather for {q}"
    
    await send_progress("Formatting weather data")
    
    location = data.get("location", {})
    current = data.get("current", {})
    condition = current.get("condition", {})
    
    result = f"""Current Weather for {location.get('name', '')}, {location.get('region', '')}, {location.get('country', '')}:
Local Time: {location.get('localtime', '')}
Temperature: {current.get('temp_c', '')}°C / {current.get('temp_f', '')}°F
Condition: {condition.get('text', '')}
Feels Like: {current.get('feelslike_c', '')}°C / {current.get('feelslike_f', '')}°F
Wind: {current.get('wind_kph', '')} kph / {current.get('wind_mph', '')} mph, direction {current.get('wind_dir', '')}
Humidity: {current.get('humidity', '')}%
Cloud Cover: {current.get('cloud', '')}%
Precipitation: {current.get('precip_mm', '')} mm / {current.get('precip_in', '')} in
UV Index: {current.get('uv', '')}
"""
    
    await send_progress("Completed")
    return result

@mcp.tool()
async def get_weather_forecast(q: str, days: int = 3) -> str:
    """
    Get weather forecast for a location.
    
    Parameters:
    - q: Location query (city name, lat/lon, IP address, US zip, UK postcode, etc.)
    - days: Number of days for forecast (1-14)
    """
    if days < 1 or days > 14:
        await send_progress("Invalid days parameter")
        return "Days parameter must be between 1 and 14"
    
    await send_progress(f"Fetching {days}-day forecast for {q}")
    data = await weather_api_request("forecast.json", {"q": q, "days": days, "aqi": "yes", "alerts": "yes"})
    
    if not data:
        await send_progress(f"Failed to get weather forecast for {q}")
        return f"Failed to get weather forecast for {q}"
    
    await send_progress("Processing forecast data")
    
    location = data.get("location", {})
    current = data.get("current", {})
    forecast = data.get("forecast", {})
    alerts = data.get("alerts", {})
    
    # Format current weather
    await send_progress("Formatting current weather")
    current_condition = current.get("condition", {})
    current_weather = f"""Current Weather for {location.get('name', '')}, {location.get('region', '')}, {location.get('country', '')}:
Local Time: {location.get('localtime', '')}
Temperature: {current.get('temp_c', '')}°C / {current.get('temp_f', '')}°F
Condition: {current_condition.get('text', '')}
Feels Like: {current.get('feelslike_c', '')}°C / {current.get('feelslike_f', '')}°F
Wind: {current.get('wind_kph', '')} kph / {current.get('wind_mph', '')} mph, direction {current.get('wind_dir', '')}
Humidity: {current.get('humidity', '')}%
"""
    
    # Format forecast days
    await send_progress("Formatting forecast days")
    forecast_days = []
    for day in forecast.get("forecastday", []):
        day_date = day.get("date", "")
        day_data = day.get("day", {})
        day_condition = day_data.get("condition", {})
        
        forecast_days.append(f"""Date: {day_date}
Min/Max Temp: {day_data.get('mintemp_c', '')}°C to {day_data.get('maxtemp_c', '')}°C / {day_data.get('mintemp_f', '')}°F to {day_data.get('maxtemp_f', '')}°F
Condition: {day_condition.get('text', '')}
Chance of Rain: {day_data.get('daily_chance_of_rain', '')}%
Max Wind: {day_data.get('maxwind_kph', '')} kph / {day_data.get('maxwind_mph', '')} mph
Avg Humidity: {day_data.get('avghumidity', '')}%
UV Index: {day_data.get('uv', '')}
""")
    
    # Format alerts if any
    await send_progress("Checking for weather alerts")
    alert_text = ""
    alert_list = alerts.get("alert", [])
    if alert_list:
        alert_items = []
        for alert in alert_list:
            alert_items.append(f"""Alert: {alert.get('headline', '')}
Category: {alert.get('category', '')}
Severity: {alert.get('severity', '')}
Urgency: {alert.get('urgency', '')}
Areas: {alert.get('areas', '')}
Effective: {alert.get('effective', '')}
Expires: {alert.get('expires', '')}
Description: {alert.get('desc', '')}
""")
        alert_text = "Weather Alerts:\n" + "\n".join(alert_items)
    
    await send_progress("Forecast data ready")
    return current_weather + "\n\nForecast:\n" + "\n".join(forecast_days) + "\n\n" + alert_text

@mcp.tool()
async def search_locations(q: str) -> str:
    """
    Search for locations using Weather API.
    
    Parameters:
    - q: Location search query (city or partial name)
    """
    await send_progress(f"Searching for locations matching '{q}'")
    data = await weather_api_request("search.json", {"q": q})
    
    if not data:
        await send_progress(f"No locations found for '{q}'")
        return f"No locations found for '{q}'"
    
    if not isinstance(data, list):
        await send_progress(f"Invalid response format: {data}")
        return f"Invalid response format for location search: {data}"
    
    if not data:
        await send_progress(f"No locations found matching '{q}'")
        return f"No locations found matching '{q}'"
    
    await send_progress(f"Found {len(data)} locations, formatting results")
    
    locations = []
    for location in data:
        locations.append(f"""{location.get('name', '')}, {location.get('region', '')}, {location.get('country', '')}
Coordinates: {location.get('lat', '')}, {location.get('lon', '')}
ID: {location.get('id', '')}
""")
    
    await send_progress("Search completed")
    return f"Found {len(locations)} locations matching '{q}':\n\n" + "\n".join(locations)

@mcp.tool()
async def get_time_zone(q: str) -> str:
    """
    Get timezone information for a location.
    
    Parameters:
    - q: Location query (city name, lat/lon, IP address, US zip, UK postcode, etc.)
    """
    await send_progress(f"Fetching timezone data for {q}")
    data = await weather_api_request("timezone.json", {"q": q})
    
    if not data:
        await send_progress(f"Failed to get timezone information for {q}")
        return f"Failed to get timezone information for {q}"
    
    await send_progress("Formatting timezone data")
    
    location = data.get("location", {})
    
    result = f"""Timezone Information for {location.get('name', '')}, {location.get('region', '')}, {location.get('country', '')}:
Timezone: {location.get('tz_id', '')}
Local Time: {location.get('localtime', '')}
Latitude: {location.get('lat', '')}
Longitude: {location.get('lon', '')}
"""
    
    await send_progress("Timezone data ready")
    return result

if __name__ == "__main__":
    print("Starting MCP Integration Server with stdio transport...")
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Error with stdio transport: {e}")
        print("Trying HTTP transport...")
        try:
            mcp.run(transport="http", port=8000)
        except Exception as e2:
            print(f"Error with HTTP transport: {e2}")
            print("Trying WebSocket transport...")
            mcp.run(transport="ws", port=8001)