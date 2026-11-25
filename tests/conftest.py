"""Shared test setup."""

import pytest

from localive import live


@pytest.fixture(autouse=True)
def _prefix_file_in_tmp(tmp_path, monkeypatch):
    """Sessions save their opening prompt for the next start: never into the checkout."""
    monkeypatch.setattr(live, "PREFIX_FILE", str(tmp_path / "brain-prefix.json"))
