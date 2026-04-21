# Root-level conftest.py — tells pytest to ignore the flat-entry loader shim
# (repo-root __init__.py) which uses a relative import that only resolves when
# Hermes loads it as _hermes_user_memory.cashew, not when pytest imports it.
collect_ignore = ["__init__.py"]
