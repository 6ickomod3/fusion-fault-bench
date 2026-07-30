from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPOSITORY_ROOT / "docs" / "dashboard.html"


class DashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.navigation_hrefs: list[str] = []
        self.headings: list[int] = []
        self.text_parts: list[str] = []
        self.style_parts: list[str] = []
        self.sections: list[tuple[str | None, str | None]] = []
        self.navigation_labels: list[str | None] = []
        self.resource_attributes: list[tuple[str, str, str]] = []
        self.inline_event_attributes: list[tuple[str, str]] = []
        self.image_alt_values: list[str | None] = []
        self.html_language: str | None = None
        self.h1_count = 0
        self.main_ids: list[str | None] = []
        self.header_count = 0
        self.footer_count = 0
        self.script_count = 0
        self.style_count = 0
        self.viewport_count = 0
        self.title_depth = 0
        self.title_text: list[str] = []
        self.style_depth = 0
        self.navigation_depth = 0
        self.skip_link_targets: list[str] = []

    @staticmethod
    def _attributes(attrs: list[tuple[str, str | None]]) -> dict[str, str | None]:
        return dict(attrs)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = self._attributes(attrs)
        element_id = attributes.get("id")
        if element_id is not None:
            self.ids.append(element_id)

        for name, value in attrs:
            if name.startswith("on"):
                self.inline_event_attributes.append((tag, name))
            if name in {"src", "srcset", "poster", "data"} and value is not None:
                self.resource_attributes.append((tag, name, value))

        if tag == "html":
            self.html_language = attributes.get("lang")
        elif tag == "header":
            self.header_count += 1
        elif tag == "main":
            self.main_ids.append(attributes.get("id"))
        elif tag == "footer":
            self.footer_count += 1
        elif tag == "nav":
            self.navigation_depth += 1
            self.navigation_labels.append(attributes.get("aria-label"))
        elif tag == "section":
            self.sections.append((attributes.get("id"), attributes.get("aria-labelledby")))
        elif tag == "script":
            self.script_count += 1
        elif tag == "style":
            self.style_count += 1
            self.style_depth += 1
        elif tag == "img":
            self.image_alt_values.append(attributes.get("alt"))
        elif tag == "meta" and attributes.get("name") == "viewport":
            self.viewport_count += 1
        elif tag == "title":
            self.title_depth += 1

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self.headings.append(level)
            if level == 1:
                self.h1_count += 1

        if tag == "a":
            href = attributes.get("href")
            if href is not None:
                self.hrefs.append(href)
                if self.navigation_depth:
                    self.navigation_hrefs.append(href)
                classes = (attributes.get("class") or "").split()
                if "skip-link" in classes:
                    self.skip_link_targets.append(href)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav":
            self.navigation_depth -= 1
        elif tag == "style":
            self.style_depth -= 1
        elif tag == "title":
            self.title_depth -= 1

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)
        if self.style_depth:
            self.style_parts.append(data)
        if self.title_depth:
            self.title_text.append(data)


def _parse_dashboard() -> tuple[DashboardParser, str]:
    source = DASHBOARD_PATH.read_text(encoding="utf-8")
    parser = DashboardParser()
    parser.feed(source)
    parser.close()
    return parser, source


def _normalized_text(parts: list[str]) -> str:
    return " ".join(" ".join(parts).split())


def test_dashboard_has_accessible_landmarks_and_heading_structure() -> None:
    parser, _ = _parse_dashboard()

    assert parser.html_language == "en"
    assert parser.header_count == 1
    assert parser.main_ids == ["main-content"]
    assert parser.footer_count == 1
    assert parser.h1_count == 1
    assert parser.viewport_count == 1
    assert parser.skip_link_targets == ["#main-content"]
    assert parser.navigation_labels == ["Dashboard sections"]
    assert parser.script_count == 0
    assert parser.style_count == 1
    assert all(
        right - left <= 1 for left, right in zip(parser.headings, parser.headings[1:], strict=False)
    )
    assert _normalized_text(parser.title_text) == "Fusion Fault Bench — Research Dashboard"

    assert len(parser.ids) == len(set(parser.ids))
    known_ids = set(parser.ids)
    for section_id, label_id in parser.sections:
        assert section_id is not None
        assert label_id is not None
        assert label_id in known_ids

    expected_navigation = {
        "#overview",
        "#architecture",
        "#evidence",
        "#m5",
        "#boundaries",
        "#next",
    }
    assert set(parser.navigation_hrefs) == expected_navigation


def test_dashboard_uses_no_external_assets_or_unsafe_inline_handlers() -> None:
    parser, source = _parse_dashboard()

    assert parser.resource_attributes == []
    assert parser.inline_event_attributes == []
    assert all(alt is not None for alt in parser.image_alt_values)
    assert "http://" not in source
    assert "https://" not in source
    assert "@import" not in source

    css = _normalized_text(parser.style_parts)
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css


def test_dashboard_internal_navigation_and_relative_links_resolve() -> None:
    parser, _ = _parse_dashboard()
    known_ids = set(parser.ids)

    for href in parser.hrefs:
        parsed = urlsplit(href)
        assert parsed.scheme == ""
        assert parsed.netloc == ""
        if href.startswith("#"):
            assert href[1:] in known_ids
            continue

        assert parsed.path
        resolved = (DASHBOARD_PATH.parent / parsed.path).resolve(strict=True)
        assert resolved.is_relative_to(REPOSITORY_ROOT)
        assert resolved.is_file()
        if parsed.fragment:
            assert parsed.fragment in known_ids

    assert "../tools/m5_release.py" in parser.hrefs


def test_dashboard_preserves_claim_safe_release_status() -> None:
    parser, source = _parse_dashboard()
    text = _normalized_text(parser.text_parts)

    if "Fusion Fault Bench reviewed evidence" in text:
        required_statements = (
            "M1–M5 have reviewed release evidence.",  # noqa: RUF001
            "M5 is published from the immutable package.",
            "Reviewed preregistered M5 outcomes",
            "The immutable package, independent reviews, deterministic document projection, "
            "and offline validation are the authority for this closeout.",
        )
        stale_status_statements = (
            "pre-outcome",
            "not released",
            "No M5 outcome",
            "Whole-revision implementation review remains next.",
            "M5's next work",
            ">Pending<",
        )
        for statement in required_statements:
            assert statement in text
        for statement in stale_status_statements:
            assert statement not in text
    else:
        required_statements = (
            "M1–M4 released.",  # noqa: RUF001
            "M5 runner and release tooling implemented; pre-outcome; not released.",
            "Local Mini profile checked, but not M5 release evidence.",
            "Release tooling is implemented; whole-revision implementation review is next.",
            "No authoritative M5 execution or scientific outcome is asserted here.",
            "A local Mini profile was checked; that check is not authoritative replay evidence.",
            "The runner and release tooling are implemented, but M5 is not released.",
            "Whole-revision implementation review remains next.",
            "M2 is implementation and local profile validation. M5 is pre-outcome.",
        )
        for statement in required_statements:
            assert statement in text

        forbidden_outcome_claims = (
            "M5 results show",
            "M5 achieved",
            "M5 release is available",
            "M5 scientific evidence proves",
        )
        for statement in forbidden_outcome_claims:
            assert statement not in text

        stale_status_statements = (
            "Release tooling is next.",
            "Release-candidate and final-package tooling remain next.",
            "next implementation target is release tooling",
            "Implement release-candidate and final-package tooling",
        )
        for statement in stale_status_statements:
            assert statement not in text

    forbidden_private_path_fragments = (
        "/Users/",
        "/home/",
        "file://",
        "C:\\",
    )
    for fragment in forbidden_private_path_fragments:
        assert fragment not in source
