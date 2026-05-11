# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Loong Ma

"""Shared utility helpers."""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_environment() -> bool:
    """Load environment variables from a local .env file if present."""
    dotenv_path = find_dotenv(filename=".env", usecwd=True)
    if not dotenv_path:
        return False

    return load_dotenv(dotenv_path=dotenv_path, override=False)
