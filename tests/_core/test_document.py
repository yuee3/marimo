# Copyright 2026 Marimo. All rights reserved.
"""The document's etags, change reporting, and publication order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from msgspec import UNSET

from marimo._core import protocol
from marimo._core.document import (
    CellSource,
    Document,
    app_config_etag,
    cell_etag,
    code_hash,
    document_etag,
    order_etag,
)
from marimo._core.errors import EditConflict, Invalid
from marimo._core.host import Host
from marimo._core.notebook import Notebook
from marimo._core.stream import Event
from tests._core.conftest import names

if TYPE_CHECKING:
    from collections.abc import Mapping

NOTEBOOK = protocol.NotebookId("n1")


def source(cell_id: str, code: str, **fields: object) -> CellSource:
    return CellSource(protocol.CellId(cell_id), code, **fields)  # type: ignore[arg-type]


HIDDEN = protocol.CellConfig(
    column=None, disabled=False, hide_code=True, expand_output=False
)


def fixture_cells() -> list[CellSource]:
    """The cells in pair's `notebook-events.sse` fixture."""
    return [
        source("c1", "import marimo as mo"),
        source("c3", "x = 21"),
        source("k7f2", "y = x * 2", config=HIDDEN),
    ]


def defs_only(codes: Mapping[str, str]) -> dict[str, protocol.Analysis]:
    """An analyzer that reports an error on every cell defining `x`."""
    out = {}
    definers = [cell_id for cell_id, code in codes.items() if "x =" in code]
    for cell_id, code in codes.items():
        errors = []
        if cell_id in definers and len(definers) > 1:
            errors.append(
                protocol.Error(
                    kind="multiple-definitions",
                    message="x is defined twice",
                    source="marimo",
                    severity="error",
                    range=None,
                    name=None,
                    traceback=None,
                    related=[],
                )
            )
        out[cell_id] = protocol.Analysis(
            defs=[code.split(" =")[0]] if " =" in code else [],
            refs=[],
            errors=errors,
        )
    return out


class TestEtags:
    def test_match_the_values_pinned_in_the_fixtures(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {"width": "full"})

        c1, c3, k7f2 = document.cells
        assert (c3.code_hash, c3.etag) == (
            "0f71456db2ea83f4",
            "5cf78587dcf22d8d",
        )
        assert (k7f2.code_hash, k7f2.etag) == (
            "db2b9ee97e1ecf4c",
            "f199b5f48ae0aabb",
        )
        assert c1.etag == "48d6363150d6281c"
        assert document.document.order_etag == "b205774f616b44e3"
        assert document.document.app_config_etag == "283513048dfeb45b"
        assert document.document.etag == "0129d119bfb96bf9"
        assert document.document.saved_etag == "0129d119bfb96bf9"

    def test_are_derived_from_their_parts(self) -> None:
        hashed = code_hash("x = 21")
        assert cell_etag(hashed, "_", HIDDEN) != cell_etag(
            hashed,
            "_",
            protocol.CellConfig(
                column=None,
                disabled=False,
                hide_code=False,
                expand_output=False,
            ),
        )
        assert cell_etag(hashed, "named", HIDDEN) != cell_etag(
            hashed, "_", HIDDEN
        )
        assert order_etag(["a", "b"]) != order_etag(["b", "a"])
        assert app_config_etag({"b": 1, "a": 2}) == app_config_etag(
            {"a": 2, "b": 1}
        )
        assert document_etag(
            order_etag(["a"]), app_config_etag({}), [hashed]
        ) != (document_etag(order_etag(["a"]), app_config_etag({}), []))

    def test_normalizes_line_endings_before_hashing(self) -> None:
        document = Document(NOTEBOOK, [source("a", "x = 1\r\ny = 2")], {})
        (cell,) = document.cells
        assert cell.code == "x = 1\ny = 2"
        assert cell.code_hash == code_hash("x = 1\ny = 2")


class TestReplace:
    def test_reports_new_changed_and_removed_cells(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {"width": "full"})
        before = document.document

        changed = document.replace(
            [
                source("c1", "import marimo as mo"),
                source("new", "z = 3"),
                source("k7f2", "y = x * 3", config=HIDDEN),
            ],
            {"width": "full"},
        )

        assert [cell.id for cell in changed.cells] == ["new", "k7f2"]
        assert changed.removed == ["c3"]
        assert changed.document is not None
        assert changed.document.order == ("c1", "new", "k7f2")
        assert changed.document.order_etag != before.order_etag
        assert changed.document.app_config_etag == before.app_config_etag
        assert changed.document.saved_etag == before.etag

    def test_a_change_that_changes_nothing_reports_nothing(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {"width": "full"})

        assert not document.replace(fixture_cells(), {"width": "full"})

    def test_an_app_config_change_moves_only_its_etag(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {})
        before = document.document

        changed = document.replace(fixture_cells(), {"width": "full"})

        assert changed.cells == []
        assert changed.removed == []
        assert changed.document is not None
        assert changed.document.order_etag == before.order_etag
        assert changed.document.app_config_etag != before.app_config_etag
        assert changed.document.etag != before.etag

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            Document(NOTEBOOK, [source("a", "1"), source("a", "2")], {})

    def test_marks_saved_state_separately(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {}, saved=False)
        assert document.document.saved_etag is None

        document.replace([source("c1", "1")], {})
        saved = document.mark_saved()

        assert saved is not None
        assert saved.saved_etag == saved.etag
        assert document.mark_saved() is None

        # A store may hold a revision other than the live one.
        other = Document(NOTEBOOK, fixture_cells(), {}, saved=False)
        behind = other.mark_saved(saved.etag)
        assert behind is not None
        assert behind.saved_etag == saved.etag != behind.etag


class TestAnalysis:
    def test_reruns_over_the_whole_document(self) -> None:
        document = Document(
            NOTEBOOK,
            [source("a", "x = 1"), source("b", "y = 2")],
            {},
            analyzer=defs_only,
        )
        a, _ = document.cells
        assert a.analysis == protocol.Analysis(defs=["x"], refs=[], errors=[])

        changed = document.replace(
            [source("a", "x = 1"), source("b", "x = 2")], {}
        )

        # Both cells re-emit: b because its code changed, a because its
        # analysis gained an error while its etag stayed put.
        assert [cell.id for cell in changed.cells] == ["a", "b"]
        new_a = changed.cells[0]
        assert new_a.etag == a.etag
        assert new_a.analysis is not None
        assert [e.kind for e in new_a.analysis.errors] == [
            "multiple-definitions"
        ]

    def test_is_null_without_an_analyzer(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {})
        assert all(cell.analysis is None for cell in document.cells)


class TestNotebook:
    def test_load_publishes_cells_then_the_document(
        self, host: Host, notebook: Notebook, log: list[Event]
    ) -> None:
        notebook.load(Document(NOTEBOOK, fixture_cells(), {"width": "full"}))

        assert names(log) == ["cell", "cell", "cell", "document"]
        snapshot = host.events(notebook_id="n1")
        assert names(snapshot) == [
            "notebook",
            "cell",
            "cell",
            "cell",
            "document",
            "ready",
        ]
        assert names(host.events(notebook_id="n1", types={"document"})) == [
            "document",
            "ready",
        ]
        # The catalog does not carry the document.
        assert "cell" not in names(host.events())

    def test_replace_publishes_what_changed(
        self, notebook: Notebook, log: list[Event]
    ) -> None:
        notebook.load(Document(NOTEBOOK, fixture_cells(), {}))
        log.clear()

        notebook.replace(
            [
                source("c1", "import marimo as mo"),
                source("c3", "x = 22"),
            ],
            {},
            request_id="edit-1",
        )

        assert names(log) == ["cell", "cell.removed", "document"]
        assert all(e.data.request_id == "edit-1" for e in log)  # type: ignore[union-attr]
        assert log[1].data.id == "k7f2"  # type: ignore[union-attr]

    def test_load_twice_is_an_error(self, notebook: Notebook) -> None:
        notebook.load(Document(NOTEBOOK, [], {}))
        with pytest.raises(ValueError):
            notebook.load(Document(NOTEBOOK, [], {}))


def put(
    cell_id: str,
    code: str,
    base_etag: str | None,
    *,
    after: str | None = None,
    before: str | None = None,
    name: str = "_",
) -> protocol.PutCell:
    return protocol.PutCell(
        base_etag=base_etag,
        cell=protocol.CellInput(id=cell_id, code=code, name=name),
        after=UNSET if after is None else after,
        before=UNSET if before is None else before,
    )


class TestApply:
    def test_creates_replaces_deletes_and_reorders_as_one_unit(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {"width": "full"})
        before = document.document
        c3 = document.cell("c3")
        k7f2 = document.cell("k7f2")
        assert c3 is not None
        assert k7f2 is not None

        applied = document.apply(
            [
                put("c2", "print('hi')", None, after="c1"),
                put("c3", "x = 22", c3.etag),
                protocol.DeleteCell(id="k7f2", base_etag=k7f2.etag),
                protocol.SetOrder(
                    base_etag=before.order_etag, order=["c3", "c2", "c1"]
                ),
                protocol.SetAppConfig(
                    base_etag=before.app_config_etag, app_config={}
                ),
            ]
        )

        assert document.order == ["c3", "c2", "c1"]
        assert [c.id for c in applied.changed.cells] == ["c3", "c2"]
        assert applied.changed.removed == ["k7f2"]
        assert applied.changed.document is not None
        assert applied.changed.document.app_config == {}
        assert set(applied.cells) == {"c2", "c3", "k7f2"}
        assert applied.cells["k7f2"] is None
        new_c3 = document.cell("c3")
        assert new_c3 is not None
        assert applied.cells["c3"] == protocol.CellEtags(
            etag=new_c3.etag, code_hash=new_c3.code_hash
        )
        assert new_c3.etag != c3.etag

    def test_a_stale_etag_applies_nothing_and_names_the_change(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {})
        before = document.document

        with pytest.raises(EditConflict) as raised:
            document.apply(
                [
                    put("new", "z = 1", None),
                    put("c3", "x = 22", "0000000000000000"),
                ]
            )

        assert raised.value.change == 1
        assert raised.value.cell == document.cell("c3")
        assert raised.value.document == before
        assert document.document == before
        assert document.cell("new") is None

    def test_creating_an_existing_cell_or_replacing_a_missing_one_conflicts(
        self,
    ) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {})

        with pytest.raises(EditConflict) as raised:
            document.apply([put("c3", "x = 1", None)])
        assert raised.value.cell == document.cell("c3")

        with pytest.raises(EditConflict) as raised:
            document.apply([put("nope", "x = 1", "*")])
        assert raised.value.cell is None

    def test_star_skips_the_check(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {})

        document.apply(
            [
                put("c3", "x = 23", "*"),
                protocol.SetOrder(base_etag="*", order=["k7f2", "c3", "c1"]),
                protocol.SetAppConfig(base_etag="*", app_config={"a": 1}),
            ]
        )

        assert document.order == ["k7f2", "c3", "c1"]
        assert document.document.app_config == {"a": 1}

    def test_placement_has_no_precondition(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {})

        document.apply([put("a", "1", None, before="c1")])
        document.apply([put("b", "2", None, after="c1")])
        document.apply([put("c", "3", None)])

        assert document.order == ["a", "c1", "b", "c3", "k7f2", "c"]

    def test_malformed_changes_are_invalid(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {})

        with pytest.raises(Invalid):
            document.apply(
                [protocol.SetOrder(base_etag="*", order=["c1", "c3"])]
            )
        with pytest.raises(Invalid):
            document.apply(
                [protocol.SetOrder(base_etag="*", order=["c1", "c3", "c3"])]
            )
        with pytest.raises(Invalid):
            document.apply([put("a", "1", None, after="nope")])
        with pytest.raises(Invalid):
            document.apply([put("a", "1", None, after="c1", before="c3")])

    def test_an_edit_that_changes_nothing_reports_nothing(self) -> None:
        document = Document(NOTEBOOK, fixture_cells(), {})
        c3 = document.cell("c3")
        assert c3 is not None

        applied = document.apply([put("c3", "x = 21", c3.etag)])

        assert not applied.changed
        assert applied.cells["c3"] == protocol.CellEtags(
            etag=c3.etag, code_hash=c3.code_hash
        )

    def test_the_notebook_publishes_an_edit_as_one_unit(
        self, notebook: Notebook, log: list[Event]
    ) -> None:
        notebook.load(Document(NOTEBOOK, fixture_cells(), {}))
        log.clear()
        k7f2 = notebook.document.cell("k7f2")  # type: ignore[union-attr]
        assert k7f2 is not None

        response = notebook.edit(
            [
                put("c2", "print('hi')", None, after="c1"),
                protocol.DeleteCell(id="k7f2", base_etag=k7f2.etag),
            ],
            request_id="edit-1",
        )

        assert names(log) == ["cell", "cell.removed", "document"]
        assert all(e.data.request_id == "edit-1" for e in log)  # type: ignore[union-attr]
        assert response.document == log[-1].data.document  # type: ignore[union-attr]
        assert set(response.cells) == {"c2", "k7f2"}
