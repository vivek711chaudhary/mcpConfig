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

@mcp.tool()
async def automate_web_interaction(url: str, instructions: str) -> str:
    """Automate web interactions using Playwright"""
    
    async def report_progress(message: str):
        await send_progress(message)
    
    payload = {
        "url": url,
        "instructions": instructions
    }
    
    await report_progress(f"Starting web automation for {url}")
    
    if instructions == "screenshot":
        try:
            await report_progress("Taking screenshot")
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response = await client.post(
                    f"{PLAYWRIGHT_MCP_URL}/automate",
                    json=payload,
                    headers={"Accept": "image/png"}
                )
                response.raise_for_status()
                await report_progress("Screenshot taken")
                return "Screenshot received (binary data)"
        except Exception as e:
            return f"Failed to get screenshot: {str(e)}"
    else:
        data = await make_api_request(
            f"{PLAYWRIGHT_MCP_URL}/automate",
            method="POST",
            json_data=payload,
            progress_callback=report_progress
        )
        
        if not data:
            return "Failed to perform web automation"
        
        await report_progress("Web automation completed")
        return data.get("text", "No text content received")

# Code Executor Endpoints
@mcp.tool()
async def execute_python_code(code: str) -> str:
    """Execute Python code in secure environment"""
    payload = {
        "code": code,
        "timeout": 10,
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

@mcp.tool()
async def get_topic_latest(topic: str) -> str:
    """Get latest information for a topic"""
    data = await make_api_request(
        f"{PLAYWRIGHT_MCP_URL}/api/topic/{topic}/latest"
    )
    
    if not data:
        return f"No latest data found for topic {topic}"
    
    # Format the response based on what the API returns
    if isinstance(data, dict):
        return f"Latest information for {topic}:\n{json.dumps(data, indent=2)}"
    else:
        return f"Latest information for {topic}:\n{data}"

@mcp.tool()
async def debug_last_request() -> str:
    """Debug the last API request that failed"""
    return f"""
To debug API requests that are timing out (MCP error -32001):

1. Check API endpoint is correct:
   - enhace-tweets-playwright → {PLAYWRIGHT_MCP_URL}/enhance-tweets-playwright
   - masa enhance → {MCP2_BASE_URL}/api/masa/enhance
   - store engagement → {PROPHET_SERVICE_URL}/api/v1/store-engagement

2. Verify request payload format:
   - For enhance-tweets-playwright: Use "tweets" list with tweet objects
   - For masa enhance: Use "query", "max_results", "enhance_top_x"
   - For playright enhance: Use "query", "max_results", "enhance_top_x" 

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

if __name__ == "__main__":
    print("Starting MCP Integration Server with stdio transport...")
    mcp.run(transport="stdio")