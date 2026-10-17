"""Native pointer positions must preserve every text candidate and delimiter."""

import pytest

from s1mb.adapters.jevvision import pointer_positions


def test_complete_pointer_record_keeps_all_151_candidates():
    ids = [10, 500, 11, 501]
    ends = []
    for i in range(151):
        ids.extend([12, 600 + i, 13])
        ends.append(len(ids) - 1)
    ids.append(14)
    assert pointer_positions(ids, [10, 11, 12, 13, 14], 151) == (len(ids) - 1, ends)


@pytest.mark.parametrize('ids', [
    [10, 11, 12, 13, 14],  # Missing a declared candidate.
    [10, 11, 12, 12, 13, 13, 14],  # Nested candidates.
    [10, 11, 12, 13, 14, 12, 13],  # Candidate after the decision.
    [11, 10, 12, 13, 12, 13, 14],  # State after the question.
])
def test_malformed_record_is_rejected(ids):
    with pytest.raises(ValueError):
        pointer_positions(ids, [10, 11, 12, 13, 14], 2)
