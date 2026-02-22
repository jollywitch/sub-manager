from __future__ import annotations

from sub_manager.error_utils import (
    exception_chain,
    format_exception_chain,
    format_exception_with_traceback,
    root_cause_summary,
)


def test_exception_chain_follows_explicit_cause() -> None:
    try:
        try:
            raise ImportError("missing dependency")
        except ImportError as exc:
            raise RuntimeError("wrapper failure") from exc
    except RuntimeError as outer:
        chain = exception_chain(outer)
        assert len(chain) == 2
        assert isinstance(chain[0], RuntimeError)
        assert isinstance(chain[1], ImportError)
        assert root_cause_summary(outer) == "ImportError: missing dependency"


def test_exception_chain_uses_context_when_cause_absent() -> None:
    try:
        try:
            raise ValueError("inner")
        except ValueError:
            raise RuntimeError("outer")
    except RuntimeError as outer:
        chain = exception_chain(outer)
        assert len(chain) == 2
        assert isinstance(chain[0], RuntimeError)
        assert isinstance(chain[1], ValueError)


def test_format_exception_with_traceback_contains_chain_and_traceback() -> None:
    try:
        raise OSError("disk error")
    except OSError as exc:
        text = format_exception_with_traceback(exc)
    assert "Exception chain:" in text
    assert "OSError: disk error" in text
    assert "Traceback:" in text
    assert "raise OSError(" in text


def test_format_exception_chain_compact_output() -> None:
    try:
        try:
            raise LookupError("root")
        except LookupError as exc:
            raise RuntimeError("top") from exc
    except RuntimeError as exc:
        chain_text = format_exception_chain(exc)
    assert chain_text == "RuntimeError: top -> LookupError: root"
