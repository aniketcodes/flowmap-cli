"""FlowMap Slack Agent - Main entry point."""

import os
import logging
from agent.bot import SlackBot
from agent.agent import Agent
from agent.server import FlowMapMCPServer
from agent.llm import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Start the FlowMap Slack Agent."""
    # Initialize components
    mcp = FlowMapMCPServer()
    llm = LLMClient(provider=os.getenv("LLM_PROVIDER", "ollama"))
    agent = Agent(mcp=mcp, llm=llm)
    bot = SlackBot(agent=agent)

    logger.info("FlowMap Slack Agent starting...")
    logger.info(f"LLM Provider: {llm.provider}")
    logger.info(f"MCP Tools: {mcp.list_tools()}")

    # Start the bot
    bot.start()


if __name__ == "__main__":
    main()
