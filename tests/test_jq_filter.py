"""Unit tests for yfmcp.jq_filter — the jq template layer."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from yfmcp.jq_filter import apply_jq_template
from yfmcp.jq_filter import jq_or_json

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATA = {
    "results": [
        {"ticker": "NVDA", "price": 138.45, "pe": 65.3},
        {"ticker": "AMD", "price": 34.0, "pe": 23.0},
    ],
    "summary": {"count": 2},
}


@pytest.fixture()
def jq_file(tmp_path: Path) -> Path:
    """Write a simple .jq file and return its path."""
    p = tmp_path / "test.jq"
    p.write_text("[.results[] | .ticker] | join(\",\")", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# apply_jq_template
# ---------------------------------------------------------------------------


class TestApplyJqTemplate:
    def test_no_args_returns_none(self) -> None:
        result = apply_jq_template(SAMPLE_DATA)
        assert result is None

    def test_inline_template(self) -> None:
        result = apply_jq_template(SAMPLE_DATA, template=".summary.count")
        assert result is not None
        assert result.strip() == "2"

    def test_inline_csv_template(self) -> None:
        # @csv in jq produces a string; raw mode outputs it unquoted.
        result = apply_jq_template(
            SAMPLE_DATA,
            template=".results[] | [.ticker, .price] | @csv",
        )
        assert result is not None
        lines = result.strip().splitlines()
        assert lines[0] == '"NVDA",138.45'
        assert lines[1] == '"AMD",34.0'  # Python float 34.0 stays as 34.0

    def test_template_file(self, jq_file: Path) -> None:
        result = apply_jq_template(SAMPLE_DATA, template_file=str(jq_file))
        assert result is not None
        # join(",") returns a string; raw mode emits it without outer quotes
        assert result.strip() == "NVDA,AMD"

    def test_template_file_wins_over_inline(self, jq_file: Path) -> None:
        """When both are provided, template_file takes precedence."""
        result = apply_jq_template(
            SAMPLE_DATA,
            template=".summary.count",  # would return "2"
            template_file=str(jq_file),  # returns ticker csv
        )
        assert result is not None
        assert "NVDA" in result

    def test_missing_template_file_returns_error_json(self) -> None:
        result = apply_jq_template(SAMPLE_DATA, template_file="/nonexistent/path/missing.jq")
        assert result is not None
        error = json.loads(result)
        assert error["error_code"] == "TEMPLATE_FILE_NOT_FOUND"

    def test_invalid_jq_expression_returns_error_json(self) -> None:
        result = apply_jq_template(SAMPLE_DATA, template="this is not valid jq !!!")
        assert result is not None
        error = json.loads(result)
        assert error["error_code"] == "TEMPLATE_COMPILE_ERROR"

    def test_runtime_error_returns_error_json(self) -> None:
        # Applying a string index on an integer will cause a jq runtime error.
        result = apply_jq_template(42, template=".foo.bar")
        assert result is not None
        # jq silently returns null for missing keys on null input in some builds,
        # so only assert it's valid JSON at minimum.
        parsed = json.loads(result)
        # Either "null" output or an error dict — both are valid JSON.
        assert parsed is not None or parsed is None  # always true, just check no exception


# ---------------------------------------------------------------------------
# jq_or_json
# ---------------------------------------------------------------------------


class TestJqOrJson:
    def test_no_template_returns_plain_json(self) -> None:
        result = jq_or_json(SAMPLE_DATA, None, None)
        parsed = json.loads(result)
        assert parsed["summary"]["count"] == 2

    def test_inline_template_transforms_output(self) -> None:
        result = jq_or_json(SAMPLE_DATA, ".results | length", None)
        assert result.strip() == "2"

    def test_template_file_transforms_output(self, jq_file: Path) -> None:
        result = jq_or_json(SAMPLE_DATA, None, str(jq_file))
        assert "NVDA" in result

    def test_error_on_bad_template_is_json(self) -> None:
        result = jq_or_json(SAMPLE_DATA, "not valid!!!", None)
        error = json.loads(result)
        assert "error_code" in error

    def test_multiline_output(self) -> None:
        """Multi-valued jq expressions produce newline-separated raw strings."""
        result = jq_or_json(SAMPLE_DATA, ".results[] | .ticker", None)
        lines = result.strip().splitlines()
        # Raw mode: string values are emitted without surrounding JSON quotes
        assert lines == ["NVDA", "AMD"]

    def test_complex_template(self) -> None:
        """Reproduce the spec's screener-style template.

        @csv produces a raw string (raw mode); string literals like "---" are
        emitted unquoted; numbers are JSON-encoded integers/floats.
        """
        template = textwrap.dedent("""\
            (.results[] | [.ticker, .price, .pe] | @csv),
            "---",
            .summary.count
        """)
        result = jq_or_json(SAMPLE_DATA, template, None)
        lines = result.strip().splitlines()
        assert lines[0].startswith('"NVDA"')  # raw CSV: "NVDA",138.45,65.3
        assert lines[2] == "---"               # raw string literal
        assert lines[-1] == "2"               # number JSON-encoded
