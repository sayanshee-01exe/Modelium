"""Test package root.

This file is load-bearing: without it pytest puts `tests/` on `sys.path`, and the
`tests/api` package then shadows the project's own top-level `api` package, so
`import api.dependencies` resolves to the test directory and fails.
"""
