from collections import namedtuple

Event = namedtuple("Event", ["index", "label"])


def pairs(xs):
    for i in range(len(xs) - 1):
        yield xs[i : i + 2]


def get_events(labels):
    """Given a sequence of labels yield the index and label where the sequence goes from one label to another.

    Returns a list of tuples where the first item is the index in the original sequence and the second item is the value at that index.

    Example:
    [0,0,0,1,1,1,0,0] -> [
        (2,0), # Because the sequence goes from being zeros to ones at index 2.
        (5,1), # Because the sequence goes from being ones to zeros at index 5.
    ]
    """
    for i, pair in enumerate(pairs(labels)):
        l1, l2 = pair
        if l1 != l2:
            yield Event(index=i, label=l1)


def dist(e0, e1):
    return abs(e0.index - e1.index)


def nearest_same_event(e0, other_events):
    # Only check events with the same label as e0.
    other_events = list(filter(lambda e1: e1.label == e0.label, other_events))
    if len(other_events) == 0:
        return None

    closest_e = other_events[0]
    for e1 in other_events[1:]:
        if e0.label == e1.label and dist(e0, e1) < dist(e0, closest_e):
            closest_e = e1
    return closest_e


# TODO: WIP!!
def soft_average_absolute_frame_difference(ground_truths, predictions):
    """Average Absoluted Frame Difference (aaFD) that also works when there are a different number of predictions and ground truths.

    For every predicted event find the closest corresponding ground truth event.
    Sum the frame differences.
    Divide by the number of ground truth events.

    Pros: Having more falsely predicted events cause the aaFD to get higher (worse)
    Cons: Having less predicted events that ground truth events causes the aaFD to approach zero (better). This is not right.
    """
    ground_truth_events = list(get_events(ground_truths))
    prediction_events = list(get_events(predictions))

    sum_differences = 0
    for event in prediction_events:
        nearest = nearest_same_event(event, ground_truth_events)
        sum_differences += dist(event, nearest)

    return sum_differences / len(ground_truth_events)
