"""Test package.

This marker matters: without it ``tests`` is only a namespace package, and a
top-level ``tests`` directory installed into site-packages by an unrelated
dependency wins the import, so ``from tests.conftest import ...`` resolves to
that library's fixtures and collection fails.
"""
