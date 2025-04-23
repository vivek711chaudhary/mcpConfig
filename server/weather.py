from typing import Any, List, Optional
import httpx
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# Initialize FastMCP server with basic configuration
mcp = FastMCP(name="mcp-integration")

# Constants with increased timeout
API_TIMEOUT = 60.0  # 60 seconds timeout for all API calls
NWS_API_BASE = "https://api.weather.gov"
MCP2_BASE_URL = "http://34.31.55.189:8080"
PLAYWRIGHT_MCP_URL = "https://playwright-mcp-620401541065.us-central1.run.app"
PROPHET_SERVICE_URL = "http://34.45.252.228:8000"
CODE_EXECUTOR_URL = "http://34.66.53.176:8002"
USER_AGENT = "mcp-integration/1.0"

async def make_api_request(
    url: str, 
    method: str = "GET", 
    json_data: Optional[dict] = None,
    params: Optional[dict] = None
) -> Optional[dict]:
    """Make an API request with extended timeout and error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params)
            else:
                response = await client.post(url, headers=headers, json=json_data)
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error for {url}: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"Error making request to {url}: {str(e)}")
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
    payload = {
        "query": query,
        "max_results": max_results,
        "enhance_top_x": enhance_top_x
    }
    if custom_instruction:
        payload["custom_instruction"] = custom_instruction
    
    data = await make_api_request(
        f"{MCP2_BASE_URL}/api/masa/enhance",
        method="POST",
        json_data=payload
    )
    
    if not data or "results" not in data:
        return "Failed to enhance tweets"
    
    results = []
    for item in data["results"]:
        tweet = item.get("original_tweet", {})
        enhanced = item.get("enhanced_content", "No enhanced content")
        results.append(
            f"Original: {tweet.get('Content', '')}\n"
            f"Enhanced: {enhanced}\n"
            f"Topics: {item.get('topics', [])}\n"
            f"Sentiment: {item.get('sentiment', 'unknown')}\n"
            "---"
        )
    
    return "\n".join(results)

@mcp.tool()
async def extract_search_terms(query: str, max_results: int = 5, enhance_top_x: int = 3) -> str:
    """Extract search terms from Twitter content"""
    payload = {
        "query": query,
        "max_results": max_results,
        "enhance_top_x": enhance_top_x
    }
    data = await make_api_request(
        f"{MCP2_BASE_URL}/api/masa/searchTerm",
        method="POST",
        json_data=payload
    )
    
    if not data or "results" not in data:
        return "Failed to extract search terms"
    
    return "\n".join([
        f"Original: {item.get('original_tweet', {}).get('Content', '')}\n"
        f"Search Term: {item.get('search_term', '')}\n"
        "---"
        for item in data["results"]
    ])

# Playwright MCP Endpoints
@mcp.tool()
async def enhance_tweets_playwright(tweets: List[dict], custom_instruction: Optional[str] = None) -> str:
    """Enhance tweets using Playwright"""
    payload = {
        "tweets": tweets,
    }
    if custom_instruction:
        payload["custom_instruction"] = custom_instruction
    
    data = await make_api_request(
        f"{PLAYWRIGHT_MCP_URL}/enhance-tweets-playwright",
        method="POST",
        json_data=payload
    )
    
    if not data or not data.get("success"):
        return "Failed to enhance tweets with Playwright"
    
    results = []
    for item in data.get("results", []):
        results.append(
            f"Original: {item.get('original_tweet', {}).get('Content', '')}\n"
            f"Enhanced: {item.get('enhanced_version', '')}\n"
            f"Research: {item.get('research', {}).get('generated_query', '')}\n"
            "---"
        )
    
    return "\n".join(results)

@mcp.tool()
async def automate_web_interaction(url: str, instructions: str) -> str:
    """Automate web interactions using Playwright"""
    payload = {
        "url": url,
        "instructions": instructions
    }
    
    if instructions == "screenshot":
        try:
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response = await client.post(
                    f"{PLAYWRIGHT_MCP_URL}/automate",
                    json=payload,
                    headers={"Accept": "image/png"}
                )
                response.raise_for_status()
                return "Screenshot received (binary data)"
        except Exception as e:
            return f"Failed to get screenshot: {str(e)}"
    else:
        data = await make_api_request(
            f"{PLAYWRIGHT_MCP_URL}/automate",
            method="POST",
            json_data=payload
        )
        
        if not data:
            return "Failed to perform web automation"
        
        return data.get("text", "No text content received")

# Code Executor Endpoints
@mcp.tool()
async def execute_python_code(code: str) -> str:
    """Execute Python code in secure environment"""
    payload = {
        "code": code,
        "timeout": 30,
        "memory_limit": "200m",
        "cpu_limit": 0.5,
        "validate_code": True
    }
    
    data = await make_api_request(
        f"{CODE_EXECUTOR_URL}/execute",
        method="POST",
        json_data=payload
    )
    
    if not data:
        return "Failed to execute code"
    
    return (
        f"Execution Time: {data.get('execution_time', 0)}s\n"
        f"Exit Code: {data.get('exit_code', -1)}\n"
        f"Validation: {data.get('validation_result', '')}\n"
        f"Output:\n{data.get('stdout', '')}\n"
        f"Errors:\n{data.get('stderr', '')}"
    )

@mcp.tool()
async def generate_and_execute(query: str) -> str:
    """Generate and execute Python code from natural language query"""
    payload = {
        "query": query,
        "timeout": 45,
        "memory_limit": "300m",
        "cpu_limit": 0.5
    }
    
    data = await make_api_request(
        f"{CODE_EXECUTOR_URL}/generate-and-execute",
        method="POST",
        json_data=payload
    )
    
    if not data:
        return "Failed to generate and execute code"
    
    return (
        f"Generated Code:\n{data.get('generated_code', '')}\n\n"
        f"Execution Time: {data.get('execution_time', 0)}s\n"
        f"Output:\n{data.get('stdout', '')}\n"
        f"Errors:\n{data.get('stderr', '')}"
    )

# Prophet Service Endpoints
@mcp.tool()
async def store_engagement_data(
    topic: str,
    platform: str,
    value: float,
    source: str = "api"
) -> str:
    """Store engagement data for a topic"""
    payload = {
        "topic": topic,
        "platform": platform,
        "timestamp": datetime.utcnow().isoformat(),
        "value": value,
        "metadata": {
            "source": source
        }
    }
    
    data = await make_api_request(
        f"{PROPHET_SERVICE_URL}/api/v1/store-engagement",
        method="POST",
        json_data=payload
    )
    
    if not data:
        return "Failed to store engagement data"
    
    return f"Status: {data.get('status', 'unknown')}\nMessage: {data.get('message', '')}"

@mcp.tool()
async def store_platform_engagements(
    topic: str,
    platform_engagements: List[dict]
) -> str:
    """Store engagement metrics from multiple platforms"""
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
    
    data = await make_api_request(
        f"{PROPHET_SERVICE_URL}/api/v1/store-platform-engagements",
        method="POST",
        json_data=payload
    )
    
    if not data:
        return "Failed to store platform engagements"
    
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
    payload = {
        "topic": topic,
        "platform": platform,
        "periods": periods,
        "frequency": "D",
        "include_history": include_history
    }
    
    data = await make_api_request(
        f"{PROPHET_SERVICE_URL}/api/v1/forecast",
        method="POST",
        json_data=payload
    )
    
    if not data:
        return "Failed to generate forecast"
    
    forecast = "\n".join(
        f"{date}: {value}" 
        for date, value in zip(data.get("forecast_dates", []), data.get("forecast_values", []))
    )
    
    history = "\n".join(
        f"{date}: {value}" 
        for date, value in zip(data.get("historical_dates", []), data.get("historical_values", []))
    )
    
    return (
        f"Forecast for {topic} on {platform}:\n{forecast}\n\n"
        f"Historical data:\n{history}"
    )

@mcp.tool()
async def get_topic_history(topic: str, platform: Optional[str] = None) -> str:
    """Get historical engagement data for a topic"""
    params = {}
    if platform:
        params["platform"] = platform
    
    data = await make_api_request(
        f"{PROPHET_SERVICE_URL}/api/v1/topics/{topic}/history",
        params=params
    )
    
    if not data:
        return f"No historical data found for topic {topic}"
    
    history = "\n".join(
        f"{item.get('timestamp', '')}: {item.get('value', 0)} "
        f"(items: {item.get('metadata', {}).get('items_count', 0)})"
        for item in data.get("data", [])
    )
    
    return (
        f"History for {topic} ({data.get('platform', 'all platforms')}):\n"
        f"{history}"
    )

# Original Weather Tools (kept for compatibility)
@mcp.tool()
async def get_weather_alerts(state: str) -> str:
    """Get weather alerts for a US state"""
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_api_request(url)
    
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    
    if not data["features"]:
        return "No active alerts for this state."
    
    alerts = []
    for feature in data["features"]:
        props = feature.get("properties", {})
        alerts.append(
            f"Event: {props.get('event', 'Unknown')}\n"
            f"Area: {props.get('areaDesc', 'Unknown')}\n"
            f"Severity: {props.get('severity', 'Unknown')}\n"
            f"Description: {props.get('description', 'No description')}\n"
            "---"
        )
    
    return "\n".join(alerts)

@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"

if __name__ == "__main__":
    print("Starting MCP Integration Server with stdio transport...")
    mcp.run(transport="stdio")