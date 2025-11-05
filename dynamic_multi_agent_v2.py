# -*- coding: utf-8 -*-
"""
Dynamic Multi-Agent System - Main Module
"""

import os
import asyncio
import time
from dotenv import load_dotenv

from orchestrator import MultiAgentOrchestrator, run_multi_agent_conversation

load_dotenv()


async def initialize_multi_agent() -> MultiAgentOrchestrator:
    """멀티 에이전트 시스템 초기화"""
    orchestrator = MultiAgentOrchestrator()
    connected_servers = []

    try:
        # 환경 변수에서 MCP 서버 목록 동적 로드
        server_index = 1
        while True:
            url_key = f"MCP_SERVER_{server_index}_URL"
            bearer_key = f"MCP_SERVER_{server_index}_AUTH_BEARER"
            name_key = f"MCP_SERVER_{server_index}_NAME"

            server_url = os.environ.get(url_key, "")
            if not server_url:
                break

            server_bearer = os.environ.get(bearer_key, "")
            server_name = os.environ.get(name_key, f"mcp{server_index}")

            print(f"Connecting to MCP Server '{server_name}': {server_url}")
            try:
                await orchestrator.connect_mcp_server(server_name, server_url, server_bearer)
                print(f"✓ Connected to {server_name}")
                connected_servers.append(server_name)
            except Exception as e:
                print(f"✗ Failed to connect to {server_name}: {e}")

            server_index += 1

        if not connected_servers:
            raise RuntimeError("No MCP servers connected. Check your .env configuration.")

        print(f"\n✅ Connected to {len(connected_servers)} MCP server(s): {', '.join(connected_servers)}")
        print("\nInitializing Multi-Agent System...")
        orchestrator.initialize_agents()
        print(f"✓ Multi-Agent System initialized\n")

        return orchestrator

    except Exception as e:
        raise RuntimeError(f"Failed to initialize multi-agent system: {e}") from e


async def main():
    print("="*70)
    print("🤖 Dynamic Multi-Agent System")
    print("="*70)

    orchestrator = None
    try:
        orchestrator = await initialize_multi_agent()
        print("\n✅ Multi-Agent System is ready!")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            q = input("\n🧑 You> ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            if not q:
                continue

            try:
                conversation_start = time.time()
                ans = await run_multi_agent_conversation(orchestrator, q)
                total_time = time.time() - conversation_start
                print(f"\n⏱️  총 소요 시간: {total_time:.2f}초")
            except Exception as e:
                print(f"\n❌ [Error] {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user.")
    except Exception as e:
        print(f"\n❌ [Fatal Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if orchestrator:
            print("\n🔌 Closing all MCP server connections...")
            try:
                await orchestrator.close_all_servers()
                print("✓ All connections closed.")
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())

