from pathlib import Path

from data_agent import create_planning_graph, plan_question, resume_clarification
from data_agent.config.paths import CONFIG_DIR


def test_root_package_only_contains_stable_entrypoints() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "data_agent"
    root_modules = {path.name for path in package_root.glob("*.py")}

    assert root_modules == {"__init__.py", "cli.py"}


def test_public_api_remains_available_after_package_refactor() -> None:
    assert callable(create_planning_graph)
    assert callable(plan_question)
    assert callable(resume_clarification)


def test_configuration_directory_is_independent_of_module_depth() -> None:
    assert (CONFIG_DIR / "normalization.yml").is_file()
    assert (CONFIG_DIR / "slot_rules.yml").is_file()
    assert (CONFIG_DIR / "access_policies.yml").is_file()
