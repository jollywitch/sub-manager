from __future__ import annotations

import traceback


def exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        if not current.__suppress_context__ and current.__context__ is not None:
            current = current.__context__
            continue
        break
    return chain


def format_exception_chain(exc: BaseException) -> str:
    chain = exception_chain(exc)
    parts: list[str] = []
    for item in chain:
        message = str(item).strip() or "<no message>"
        parts.append(f"{type(item).__name__}: {message}")
    return " -> ".join(parts)


def root_cause_summary(exc: BaseException) -> str:
    chain = exception_chain(exc)
    root = chain[-1] if chain else exc
    message = str(root).strip() or "<no message>"
    return f"{type(root).__name__}: {message}"


def format_exception_with_traceback(exc: BaseException) -> str:
    chain_text = format_exception_chain(exc)
    traceback_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).rstrip()
    if not traceback_text:
        return chain_text
    return f"Exception chain: {chain_text}\nTraceback:\n{traceback_text}"
