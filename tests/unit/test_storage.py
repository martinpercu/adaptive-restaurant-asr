from __future__ import annotations

import pytest

from ars.storage import LocalStorage


def test_roundtrip(tmp_path):
    st = LocalStorage(tmp_path)
    st.put("a/b/c.txt", b"hello")
    assert st.exists("a/b/c.txt")
    assert st.get("a/b/c.txt") == b"hello"


def test_list_prefix(tmp_path):
    st = LocalStorage(tmp_path)
    st.put("clean/es/x.wav", b"1")
    st.put("clean/es/y.wav", b"2")
    st.put("clean/en/z.wav", b"3")
    assert st.list("clean/es/") == ["clean/es/x.wav", "clean/es/y.wav"]
    assert st.list("clean/") == ["clean/en/z.wav", "clean/es/x.wav", "clean/es/y.wav"]


def test_missing_prefix_returns_empty(tmp_path):
    st = LocalStorage(tmp_path)
    assert st.list("nope/") == []


def test_absolute_path_rejected(tmp_path):
    st = LocalStorage(tmp_path)
    with pytest.raises(ValueError):
        st.put("/etc/passwd", b"x")


def test_escape_root_rejected(tmp_path):
    st = LocalStorage(tmp_path)
    with pytest.raises(ValueError):
        st.get("../../secret")


def test_url_is_file_uri(tmp_path):
    st = LocalStorage(tmp_path)
    st.put("a.txt", b"1")
    assert st.url("a.txt").startswith("file://")
