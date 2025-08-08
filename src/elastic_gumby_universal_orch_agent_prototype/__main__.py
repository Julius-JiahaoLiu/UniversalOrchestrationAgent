"""
Entry point for the elastic_gumby_universal_orch_agent_prototype package.

This allows the package to be executed as:
    python -m elastic_gumby_universal_orch_agent_prototype

This is the standard Python way to make packages executable.
"""

import sys
from colorama import Style, Fore
from .agent_main import AgentMainInterface

if __name__ == "__main__":
    try:
        agent = AgentMainInterface()
        agent.run()
    except Exception as e:
        print(f"{Fore.RED}Fatal error initializing UTOA: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
