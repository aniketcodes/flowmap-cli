"""FlowMap Agent - Slack Bot with MCP integration."""

import os
import re
import sys
import time
import logging
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_bolt import App
from agent.slack_client import SlackClient
from agent.slack_mcp_client import SlackMCPClient
from agent.agent import Agent
from agent.server import FlowMapMCPServer
from agent.llm import LLMClient
from agent.security import TTLCache
from agent.bot import _md_to_slack

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_bot")

# Initialize components
logger.info("Initializing components...")
slack_client = SlackClient()
mcp = FlowMapMCPServer()
llm = LLMClient(provider="ollama")

# Connect to Slack MCP server for hackathon compliance
logger.info("Connecting to Slack MCP server...")
slack_mcp_client = SlackMCPClient(server_url="http://localhost:13080/sse")
slack_mcp_connected = False

try:
    import asyncio
    slack_mcp_connected = asyncio.get_event_loop().run_until_complete(slack_mcp_client.connect())
    if slack_mcp_connected:
        logger.info("Connected to Slack MCP! Tools: %d", len(slack_mcp_client.list_tools()))
    else:
        logger.warning("Could not connect to Slack MCP server")
except Exception as e:
    logger.warning("Slack MCP connection failed: %s", e)

agent = Agent(mcp=mcp, llm=llm, slack_client=slack_client, slack_mcp_client=slack_mcp_client if slack_mcp_connected else None)
logger.info("Agent ready with %d tools", len(agent._get_tool_definitions()))

# Create Bolt app (for API calls)
app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")
)

# Create Socket Mode client
client = SocketModeClient(app_token=os.getenv("SLACK_APP_TOKEN"))

# Event deduplication — atomic via contains_and_add
_seen_events = TTLCache(ttl=600)


def process_mention(event, client):
    """Process mention in background thread with streaming."""
    message = event.get("text", "")
    user = event.get("user", "")
    channel = event.get("channel", "")
    event_id = event.get("ts", "")
    thread_ts = event.get("thread_ts", event_id)

    # Atomic dedup
    if _seen_events.contains_and_add(event_id):
        logger.info("duplicate_event event_id=%s", event_id)
        return

    logger.info("mention user=%s channel=%s message=%.200s", user, channel, message)

    clean_message = re.sub(r"<@[A-Z0-9]+>", "", message).strip()

    if not clean_message:
        response = "Hi! I can help you diagnose production issues. Ask me something about your codebase!"
        app.client.chat_postMessage(channel=channel, text=response, thread_ts=thread_ts)
        return

    # Post placeholder and stream progress
    from agent.streaming import StreamingMessage
    sm = StreamingMessage(
        client=app.client,
        channel=channel,
        initial="🤔 Analyzing your query...",
        thread_ts=thread_ts,
    )
    placeholder_ts = sm.send()

    # Fallback: if placeholder failed, just get response and post directly
    if not placeholder_ts:
        logger.warning("Placeholder failed, falling back to direct post")
        try:
            response = agent.diagnose(clean_message)
        except Exception as e:
            logger.error("diagnosis_failed error=%s", e, exc_info=True)
            response = f"Error: {e}"
        try:
            app.client.chat_postMessage(channel=channel, text=_md_to_slack(response), thread_ts=thread_ts)
        except Exception as e:
            logger.error("Fallback post failed: %s", e)
        return

    # Streaming path
    def update_progress(text):
        sm.update(text)

    logger.info("Running diagnosis...")
    try:
        response = agent.diagnose(clean_message, on_progress=update_progress)
        logger.info("Response: %.500s", response)
    except Exception as e:
        logger.error("diagnosis_failed error=%s", e, exc_info=True)
        response = f"Error: {e}"

    sm.finish(_md_to_slack(response))
    logger.info("Sent!")


def handle_request(client, request):
    if request.type == "events_api":
        event = request.payload.get("event", {})
        if event.get("type") == "app_mention":
            # Acknowledge IMMEDIATELY before processing
            response = SocketModeResponse(envelope_id=request.envelope_id)
            client.send_socket_mode_response(response)

            # Start background processing thread
            thread = threading.Thread(
                target=process_mention,
                args=(event, client),
                daemon=True
            )
            thread.start()
        else:
            # Acknowledge non-mention events immediately
            response = SocketModeResponse(envelope_id=request.envelope_id)
            client.send_socket_mode_response(response)


# Register listener
client.socket_mode_request_listeners.append(handle_request)

# Connect and run
logger.info("Connecting to Slack...")
client.connect()
logger.info("Connected! Bot is ready. Mention @FlowMap Agent in Slack.")

# Keep alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("Shutting down...")
    client.disconnect()
    if slack_mcp_connected:
        asyncio.get_event_loop().run_until_complete(slack_mcp_client.disconnect())
