from contextlib import nullcontext as does_not_raise

import pytest

from papersai.utils import load_paper_as_context


@pytest.mark.parametrize(
    "paper_id, exception",
    [
        pytest.param(
            "2301.08210",
            does_not_raise(),
            id="valid_id",
        ),
        pytest.param(None, pytest.raises(AssertionError), id="No paper id"),
        pytest.param(
            "4242.424242",
            pytest.raises(ValueError),
            id="invalid paper id",
        ),
    ],
)
def test_context(paper_id: str, exception: Exception) -> None:
    with exception:  # type: ignore[attr-defined]
        context = load_paper_as_context(paper_id=paper_id)
        assert len(context) > 0
