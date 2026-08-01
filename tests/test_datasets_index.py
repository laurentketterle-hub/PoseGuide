"""Tests for public pose datasets index."""
import os
import pytest


DATASETS_MD = os.path.join(
    os.path.dirname(__file__), '..', 'docs', 'DATASETS.md'
)


def test_datasets_md_exists():
    """Verify DATASETS.md file exists."""
    assert os.path.isfile(DATASETS_MD), f"{DATASETS_MD} not found"


def test_datasets_md_non_empty():
    """Verify DATASETS.md has content."""
    with open(DATASETS_MD, 'r', encoding='utf-8') as f:
        content = f.read()
    assert len(content) > 500, f"DATASETS.md too short: {len(content)} chars"


def test_datasets_table_has_rows():
    """Verify DATASETS.md contains the dataset table with at least 8 rows."""
    with open(DATASETS_MD, 'r', encoding='utf-8') as f:
        content = f.read()
    # Count table rows (lines starting with | after the header separator)
    table_rows = [l for l in content.split('\n') if l.startswith('|') and '|---' not in l]
    # Header row included, so >= 9 (1 header + 8 data rows)
    assert len(table_rows) >= 9, f"Expected >=9 table rows, got {len(table_rows)}"


def test_license_column_present():
    """Verify license information is included."""
    with open(DATASETS_MD, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'CC-BY' in content or 'CC0' in content or 'BSD' in content, \
        "No license info found in DATASETS.md"


def test_url_column_present():
    """Verify URL references are included."""
    with open(DATASETS_MD, 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'https://' in content, "No URLs found in DATASETS.md"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
