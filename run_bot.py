"""FlowMap Agent - Slack Bot with MCP integration."""

import os
os.environ['PYTHONUNBUFFERED'] = '1'
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
from agent.actions import ActionPlanner, ActionRegistry, ActionExecutor, build_action_blocks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")])
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
logger.info("Agent ready with %d tools", len(agent._build_openai_tools()))

# Create Bolt app (for API calls)
app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")
)

# Action planner components
action_planner = ActionPlanner(llm=llm)
action_registry = ActionRegistry(ttl=3600)
action_executor = ActionExecutor(client=app.client, agent=agent, flowmap=mcp)

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
        # If this is a thread reply, include previous conversation as context
        query = clean_message
        if thread_ts != event_id:
            thread_history = get_thread_history(channel, thread_ts)
            if thread_history:
                context = "\n".join(
                    [f"- {msg.get('text', '')}" for msg in thread_history[:10]]
                )
                query = (
                    f"Previous conversation in this thread:\n{context}\n\n"
                    f"New question: {clean_message}"
                )
                logger.info("Thread context included: %d messages", len(thread_history))
        
        response = agent.diagnose(query, on_progress=update_progress)
        logger.info("Response: %.500s", response)
    except Exception as e:
        logger.error("diagnosis_failed error=%s", e, exc_info=True)
        response = f"Error: {e}"

    logger.info("PRE-CHECK: registry=%s placeholder=%s", action_registry is not None, placeholder_ts is not None)

    sm.finish(_md_to_slack(response))
    logger.info("sm.finish completed")

    # Build action blocks and post as a separate message
    if action_registry is not None and placeholder_ts is not None:
        logger.info("ENTERING action block")
        try:
            # Use ActionPlanner to recommend actions based on diagnosis + user question
            actions = action_planner.recommend(response, user_question=clean_message)
            logger.info("Planner recommended actions: %s", [a.get('id') for a in actions])
            action_registry.store(placeholder_ts, actions, diagnosis=response)
            logger.info("Stored actions in registry")
            action_blocks = build_action_blocks(response, actions, placeholder_ts)
            logger.info("Built action_blocks: %s", action_blocks)
            logger.info("Posting action buttons: channel=%s thread_ts=%s blocks=%d", channel, placeholder_ts, len(action_blocks))
            
            # Add delay to avoid Socket Mode serialization issues
            time.sleep(1)
            
            result = app.client.chat_postMessage(
                channel=channel,
                thread_ts=placeholder_ts,
                text="Recommended Actions",
                blocks=action_blocks,
            )
            logger.info("POSTED: ok=%s result=%s", result.get('ok'), result)
            if not result.get('ok'):
                logger.error("Slack API rejected button post: %s", result.get('error'))
        except Exception as e:
            logger.error("FAILED to post action buttons: %s", e, exc_info=True)

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

    elif request.type == "interactive":
        payload = request.payload
        if payload.get("type") == "block_actions":
            # Acknowledge immediately
            response = SocketModeResponse(envelope_id=request.envelope_id)
            client.send_socket_mode_response(response)

            # Handle button clicks in background thread
            thread = threading.Thread(
                target=handle_button_click,
                args=(payload,),
                daemon=True
            )
            thread.start()
        else:
            response = SocketModeResponse(envelope_id=request.envelope_id)
            client.send_socket_mode_response(response)


def get_thread_history(channel, thread_ts, max_messages=10):
    """Fetch thread history from Slack, excluding bot messages.
    
    Uses thread_ts (root message) to fetch all replies in the thread.
    """
    try:
        result = app.client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=max_messages,
        )
        messages = result.get("messages", [])
        return [
            {"user": msg.get("user", ""), "text": msg.get("text", "")}
            for msg in messages
            if msg.get("user") != "U0BG204VAQY"
        ]
    except Exception as e:
        logger.warning("Failed to fetch thread history: %s", e)
        return []


def handle_button_click(payload):
    """Handle action button clicks with thinking placeholder."""
    actions = payload.get("actions", [])
    if not actions:
        return

    action = actions[0]
    # action_id format: "show_history|<placeholder_ts>" — use this to look up diagnosis
    action_id_raw = action.get("action_id", "")
    parts = action_id_raw.split("|", 1)
    action_id = parts[0]
    stored_ts = parts[1] if len(parts) > 1 else ""

    container = payload.get("container", {})
    message_ts = container.get("message_ts", "")
    channel_id = payload.get("channel", {}).get("id", "")

    # Thread root ts for fetching thread history
    thread_root_ts = payload.get("message", {}).get("thread_ts", "") or message_ts

    if not action_id or not message_ts:
        logger.warning("Button click missing action_id or message_ts")
        return

    logger.info("action_click action_id=%s stored_ts=%s message_ts=%s", action_id, stored_ts, message_ts)

    # Post thinking placeholder
    thinking_ts = None
    try:
        thinking_msg = app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=message_ts,
            text=":hourglass_flowing_sand: Processing action...",
        )
        thinking_ts = thinking_msg.get("ts")
    except Exception as e:
        logger.error("Failed to post thinking placeholder: %s", e)

    # Get diagnosis using stored_ts (the placeholder ts where it was stored)
    lookup_ts = stored_ts or message_ts
    diagnosis = action_registry.get_diagnosis(lookup_ts) if action_registry else "Unknown issue"
    logger.info("diagnosis_lookup ts=%s diagnosis=%.100s", lookup_ts, diagnosis)

    # Get thread history using thread root ts
    thread_history = get_thread_history(channel_id, thread_root_ts)
    logger.info("thread_history fetched=%d messages", len(thread_history))

    # Execute action
    result = action_executor.execute(action_id, message_ts, channel_id, diagnosis,
                                     thread_history=thread_history)
    logger.info("action_result action_id=%s result=%s", action_id, result)

    # Update thinking placeholder with result
    if thinking_ts:
        try:
            from agent.bot import _md_to_slack
            app.client.chat_update(
                channel=channel_id,
                ts=thinking_ts,
                text=_md_to_slack(f":white_check_mark: {result}"),
            )
        except Exception as e:
            logger.error("Failed to update thinking placeholder: %s", e)
    else:
        # Fallback: post result as new message if placeholder failed
        try:
            app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=message_ts,
                text=f":white_check_mark: {result}",
            )
        except Exception as e:
            logger.error("Failed to post action result: %s", e)


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
