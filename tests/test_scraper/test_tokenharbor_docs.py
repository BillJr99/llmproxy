"""Token Harbor docs scraper: freeRows parsing and fail-soft degradation."""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

from scripts.sources.docs.tokenharbor import TokenHarborDocs

EXPECTED = [
    "tokenharbor/deepseek-v4.1-flash:free",
    "tokenharbor/deepseek-v4-flash:free",
    "tokenharbor/mimo-v2.5:free",
]


def _html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "tokenharbor_models.html").read_text()


@responses.activate
def test_free_models_from_page(fixtures_dir: Path):
    responses.add(responses.GET, TokenHarborDocs.url, body=_html(fixtures_dir),
                  status=200, content_type="text/html")
    evs = TokenHarborDocs().fetch()
    assert [e.model_id for e in evs] == EXPECTED
    for e in evs:
        assert e.is_free is True
        assert e.confidence == "high"
        assert e.source == "tokenharbor"


def test_rejects_rows_that_fail_the_cross_check(fixtures_dir: Path):
    """The fixture seeds two traps inside freeRows: a row that bills despite
    isFree, and a row without the ":free" suffix. Neither may be published."""
    ids = {e.model_id for e in TokenHarborDocs().parse(_html(fixtures_dir))}
    assert "tokenharbor/not-actually-free:free" not in ids
    assert "tokenharbor/no-suffix" not in ids
    assert ids == set(EXPECTED)


def test_ids_are_prefixed_with_provider_key(fixtures_dir: Path):
    evs = TokenHarborDocs().parse(_html(fixtures_dir))
    assert all(e.model_id.startswith("tokenharbor/") for e in evs)


def test_free_suffix_convention_preserved(fixtures_dir: Path):
    """Token Harbor bills by id suffix, so the suffix must survive into
    believed_free for endpoint_probe to maintain the list later."""
    evs = TokenHarborDocs().parse(_html(fixtures_dir))
    assert all(e.model_id.endswith(":free") for e in evs)


@pytest.mark.parametrize("html", [
    "",
    "<html><body>no payload here</body></html>",
    'x "freeRows":[{"surface":"a:free",',        # truncated array
    'x "freeRows":"not-a-list"',
    'x "freeRows":[{surface:not-json}]',
])
def test_malformed_pages_degrade_to_no_evidence(html: str):
    """A page redesign must mean "no opinion", never an exception. Docs scrapers
    emit positives only, so an empty result causes no removals."""
    assert TokenHarborDocs().parse(html) == []
