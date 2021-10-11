from edesdetectrl import environments


def test_safe_balanced_accuracy():
    assert environments.safe_balanced_accuracy([0,0,0,0], [1,1,1,1]) == -1
    assert environments.safe_balanced_accuracy([0,0,0,0], [1,1,1,0]) == -0.5
    assert environments.safe_balanced_accuracy([0,0,0,0], [1,1,0,0]) == 0
    assert environments.safe_balanced_accuracy([0,0,0,0], [1,0,0,0]) == 0.5
    assert environments.safe_balanced_accuracy([0,0,0,0], [0,0,0,0]) == 1

    assert environments.safe_balanced_accuracy([0,1,0,1], [0,1,0,1]) == 1
    assert environments.safe_balanced_accuracy([0,1,0,1], [1,0,1,0]) == -1
    assert environments.safe_balanced_accuracy([0,1,0,1], [0,0,0,0]) == 0