import pandas as pd
import cv2 as cv
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def video_as_numpy(filename, videos_path):
    cap = cv.VideoCapture(videos_path + filename + ".avi")
    all_frames = []

    ret, frame = cap.read()
    while ret:
        # It is important that we append the frame before we read the next one, because
        # at the end of the video 'frame' will be None. The loop will automatically end
        # at this point because 'ret' will be False.
        all_frames.append(frame)
        ret, frame = cap.read()

    cap.release()

    all_frames = np.mean(np.array(all_frames), axis=3)
    return all_frames


def next_maximum_diff(x, start_i):
    increasing = np.gradient(gaussian_filter(x, sigma=2)) < 0

    max_diff_i = start_i
    for i in range(start_i + 1, len(x)):
        if increasing[i]:
            break
        else:
            max_diff_i = i

    return max_diff_i


def prev_maximum_diff(x, start_i):
    increasing = np.gradient(gaussian_filter(x, sigma=2)) >= 0

    max_diff_i = start_i
    for i in range(start_i - 1, 0, -1):
        if increasing[i]:
            break
        else:
            max_diff_i = i

    return max_diff_i


def label_more_frames(x, ed_i, es_i, weight=0.75):
    """
    Grab some frames before first keyframe (either ED or ES) where we are sure of the phase.

    We can be relatively sure of the phase by looking at the difference between one of the
    keyframes (either ED or ES) and all other frames. The frames leading up to the previous
    frame with the most difference from the first keyframe will have the same phase as the
    keyframe, and likewise the next frame with the most difference from the last keyframe
    will have the opposite phase as the keyframe.

    Weight is used to ensure that we can be certain that the new labels are correct. A
    higher weight means we will look further into the "unknown", further out towards the
    previous or next maximum difference from a keyframe.
    """
    ed_i_diff = [np.sum((x[i] - x[ed_i]) ** 2) for i in range(x.shape[0])]
    es_i_diff = [np.sum((x[i] - x[es_i]) ** 2) for i in range(x.shape[0])]

    # Either ED is labeled first, or ES is. The code logic is the same, but different
    # labels have to be returned -- i.e.: it's almost copy-paste in the two clauses below.
    if ed_i < es_i:
        some_before_ed_i = int(
            prev_maximum_diff(ed_i_diff, ed_i) * weight + ed_i * (1 - weight)
        )
        some_after_es_i = int(
            next_maximum_diff(es_i_diff, es_i) * weight + es_i * (1 - weight)
        )
        frames = np.arange(some_before_ed_i, some_after_es_i + 1)
        labels = (
            ["diastole"] * (ed_i - some_before_ed_i + 1)
            + ["systole"] * (es_i - ed_i + 1)
            + ["diastole"] * (some_after_es_i - es_i + 1)
        )
        return (frames, labels)
    else:
        some_before_es_i = int(
            prev_maximum_diff(es_i_diff, es_i) * weight + es_i * (1 - weight)
        )
        some_after_ed_i = int(
            next_maximum_diff(ed_i_diff, ed_i) * weight + ed_i * (1 - weight)
        )
        frames = np.arange(some_before_es_i, some_after_ed_i + 1)
        labels = (
            ["systole"] * (es_i - some_before_es_i + 1)
            + ["diastole"] * (ed_i - es_i + 1)
            + ["systole"] * (some_after_ed_i - ed_i + 1)
        )
        return (frames, labels)


class EchoNetDataset:
    def __init__(
        self,
        processed_labels_csv_file="processed_labels.csv",
        videos_path="/home/magnus/research/data/EchoNet-Dynamic/Videos/",
    ):
        self.df = pd.read_csv(processed_labels_csv_file)
        self.num_videos = len(self.df)
        self.videos_path = videos_path

    def get_video_and_labels(self, index):
        video_df = self.df.iloc[index]
        video = video_as_numpy(video_df["FileName"], self.videos_path)
        ed_frame_i, es_frame_i = video_df["ED_Frame"], video_df["ES_Frame"]
        frames, labels = label_more_frames(video, ed_frame_i, es_frame_i, weight=0.75)

        return video[frames], labels


def test_it():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    echonet_dataset = EchoNetDataset()
    x, labels = echonet_dataset.get_video_and_labels(1000)

    fig, ax = plt.subplots()
    im = ax.imshow(x[0])
    an = ax.annotate("", (10, 10), c="w")

    def animate(i):
        im.set_data(x[i])
        an.set_text(labels[i])

    anim = FuncAnimation(fig, animate, frames=x.shape[0], interval=50)
    plt.show()


# test_it()
