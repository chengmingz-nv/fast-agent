#!/usr/bin/env python3
"""
MCP server that provides file I/O operations: create_file, read_file, and write_file.
This server allows tools for basic file system operations.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="fileio.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(
    name="FileIO-MCP-Server",
    instructions="This server provides tools for basic file I/O operations",
)


@app.tool(
    name="setup_environment",
    description="setup environment for testing on a specific IP address",
)
def setup_environment(ip: str = "localhost") -> str:
    """
    Setup environment for test.

    Args:
        ip: IP address of the server

    Returns:
        Success or error message
    """
    logger.info(rf"Setup Environment {ip}")
    return rf"Setup Environment {ip}"


if __name__ == "__main__":
    import sys

    # Detect if we should use stdio or sse transport
    transport = "stdio"
    if len(sys.argv) > 1 and sys.argv[1] == "--sse":
        logger.info("Using SSE transport")
        transport = "sse"
        host = "localhost"
        port = 8000
        if len(sys.argv) > 2:
            host = sys.argv[2]
        if len(sys.argv) > 3:
            port = int(sys.argv[3])
        print(f"Starting FileIO MCP Server with SSE transport on {host}:{port}")
        app.run(transport=transport, host=host, port=port)
    else:
        # Run with stdio transport by default
        logger.info("Starting FileIO MCP Server with stdio transport")
        app.run(transport=transport)
