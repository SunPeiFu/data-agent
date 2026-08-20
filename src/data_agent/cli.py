"""Backward-compatible module entrypoint for ``python -m data_agent.cli``."""

from data_agent.interfaces.cli import main


if __name__ == "__main__":
    main()
