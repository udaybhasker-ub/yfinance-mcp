"""jq template filtering layer for MCP tool responses.

Callers can supply an inline jq expression via ``template``.
If no template is provided the helpers behave as no-ops and the caller should
fall back to serialising raw JSON with :func:`~yfmcp.utils.dump_json`.
"""

from __future__ import annotations

import json

from yfmcp.utils import create_error_response
from yfmcp.utils import dump_json

try:
    import jq as _jq  # type: ignore[import-untyped]

    _JQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JQ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Field description constants – imported by server.py so the MCP schema
# reflects the contract in one place.
# ---------------------------------------------------------------------------

TEMPLATE_FIELD_DESCRIPTION: str = (
    "Optional jq expression to transform the JSON response "
    "(e.g. '.results[] | .symbol'). "
    "Ignored when the tool returns an image."
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def apply_jq_template(
    data: object,
    template: str | None = None,
) -> str | None:
    """Apply a jq expression to *data* and return the rendered text.

    Returns
    -------
    ``None``
        *template* was not provided — the caller
        should serialise *data* as plain JSON.
    str
        The jq output text (may be multi-line for multi-valued expressions).
        On any failure (missing file, compile error, runtime error) the
        returned string is a structured error JSON matching the project's
        ``create_error_response`` format.
    """
    if not template:
        return None

    if not _JQ_AVAILABLE:  # pragma: no cover
        return create_error_response(
            "The 'jq' package is not installed. Add it to your environment: pip install jq",
            error_code="DEPENDENCY_ERROR",
        )

    expr = template

    # Compile the jq program.
    try:
        program = _jq.compile(expr)
    except ValueError as exc:
        return create_error_response(
            f"Invalid jq expression: {exc}",
            error_code="TEMPLATE_COMPILE_ERROR",
            details={"expression": expr[:500], "exception": str(exc)},
        )

    # Execute and return output in raw mode (equivalent to jq -r):
    # string values are emitted unquoted; all other types are JSON-encoded.
    try:
        values = program.input(value=data).all()
    except Exception as exc:
        return create_error_response(
            f"jq template execution failed: {exc}",
            error_code="TEMPLATE_EXECUTION_ERROR",
            details={"expression": expr[:500], "exception": str(exc)},
        )

    lines: list[str] = []
    for v in values:
        if isinstance(v, str):
            lines.append(v)
        else:
            lines.append(json.dumps(v, ensure_ascii=False, default=str))
    return "\n".join(lines)


def jq_or_json(
    data: object,
    template: str | None,
) -> str:
    """Serialise *data*, optionally through a jq template.

    This is the standard return helper for MCP tool success paths:

    .. code-block:: python

        return jq_or_json(result, template)

    When no template arguments are provided it behaves identically to
    ``dump_json(data)``.  On template errors it returns the structured error
    JSON produced by :func:`apply_jq_template`.
    """
    if not template:
        return dump_json(data)
    result = apply_jq_template(data, template=template)
    assert result is not None, "apply_jq_template must not return None here"  # noqa: S101
    return result
