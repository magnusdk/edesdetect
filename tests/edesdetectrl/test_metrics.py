import edesdetectrl.metrics as metrics
from edesdetectrl.metrics import Event


def test_pairs():
    assert [[1, 2], [2, 3], [3, 4]] == list(metrics.pairs([1, 2, 3, 4]))
    assert [] == list(metrics.pairs([]))
    assert [] == list(metrics.pairs([1]))
    assert [[1, 2]] == list(metrics.pairs([1, 2]))


def test_get_events():
    assert [(2, 0), (4, 1)] == list(metrics.get_events([0, 0, 0, 1, 1, 0, 0]))
    assert [(0, 0), (1, 1), (2, 0)] == list(metrics.get_events([0, 1, 0, 1]))
    assert [] == list(metrics.get_events([0, 0, 0, 0, 0]))
    assert [] == list(metrics.get_events([]))


def test_nearest_same_event():
    e0 = Event(5, 0)
    assert None == metrics.nearest_same_event(e0, [])
    assert (0, 0) == metrics.nearest_same_event(e0, [Event(0, 0)])
    assert None == metrics.nearest_same_event(e0, [Event(0, 1)])
    assert (5, 0) == metrics.nearest_same_event(
        e0, [Event(0, 0), Event(2, 1), Event(5, 0)]
    )
    assert (7, 0) == metrics.nearest_same_event(
        e0, [Event(0, 0), Event(2, 1), Event(7, 0)]
    )
