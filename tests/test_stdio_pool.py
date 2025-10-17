import asyncio

import pytest

from usb_agents.tool_client import StdioToolClient


@pytest.mark.asyncio
async def test_stdio_concurrency_pool():
    client = StdioToolClient(concurrency=2)
    await client.warmup(2)

    async def call():
        return await client.call_tool("fs_read", {"path": "README.md"}, None)

    results = await asyncio.gather(call(), call())
    assert all("content" in result for result in results)
    stats = await client.collect_stats()
    assert stats["wait_ms"]["count"] == 2
    assert stats["wait_ms"]["max"] >= 0.0
    await client.aclose()
