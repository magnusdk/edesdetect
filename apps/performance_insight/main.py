import apps.performance_insight.layout as layout
import apps.performance_insight.scores_preprocessor as scores_preprocessor
import apps.performance_insight.util as util
import coax
import edesdetectrl.dataloaders.echonet as echonet
import edesdetectrl.model as model
import PySimpleGUI as sg
from apps.performance_insight.events import dispatch_on, handle_event
from apps.performance_insight.layout import (
    EVALUATION_CANVAS,
    VIDEO,
    VIDEO_SELECTOR,
    VIDEO_SELECTOR_FIRST_BUTTON,
    VIDEO_SELECTOR_LAST_BUTTON,
    VIDEO_SELECTOR_NEXT_BUTTON,
    VIDEO_SELECTOR_PREVIOUS_BUTTON,
    VIDEO_SELECTOR_SORT_BY,
)
from edesdetectrl.config import config
from edesdetectrl.environments.binary_classification import EDESClassificationBase_v0
from edesdetectrl.util.functional import chainl

SORT_OPTION_FILENAME = "filename"
SORT_OPTION_BEST = "best"
SORT_OPTION_WORST = "worst"


def video_selector_file_options_sort_fn(sort_order):
    if sort_order == SORT_OPTION_FILENAME:
        return lambda video: video.filename
    elif sort_order == SORT_OPTION_BEST:
        # Sort by negative perf, because lower perf should be sorted last
        # Unevaluated videos are sorted last, hence infinity
        return lambda video: -video.perf if video.perf is not None else float("inf")
    elif sort_order == SORT_OPTION_WORST:
        # Unevaluated videos are sorted last, hence infinity
        return lambda video: video.perf if video.perf is not None else float("inf")


class DropDownOption:
    def __init__(self, value, label):
        self.value = value
        self.label = label

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return str((self.value, self.label))


class VideoFileListItem:
    def __init__(self, filename, perf) -> None:
        self.filename = filename
        self.perf = perf

    def __str__(self) -> str:
        perf = f"{self.perf:.2f}" if self.perf is not None else "unevaluated"
        return f"{self.filename} ({perf})"


sort_options = [
    DropDownOption(SORT_OPTION_BEST, "Best"),
    DropDownOption(SORT_OPTION_WORST, "Worst"),
    DropDownOption(SORT_OPTION_FILENAME, "Filename"),
]

filenames = echonet.get_filenames(config["data"]["filelist_path"], split="TEST")
pre_processed_scores = scores_preprocessor.get_pre_processed_scores()
video_selector_file_options = [
    VideoFileListItem(filename, pre_processed_scores.get(filename, None))
    for filename in filenames
]
video_selector_file_options.sort(
    key=video_selector_file_options_sort_fn(sort_options[0].value)
)

get_video = util.video_getter()
env = EDESClassificationBase_v0()
q = coax.Q(model.get_func_approx(env), env)
q.params = coax.utils.load(config["data"]["trained_params_path"])

## Miscellaneous handlers
def handle_video_file_selected(selected_video, window):
    env.seq_and_labels = get_video(selected_video.filename)
    trajectory = env.generate_trajectory_using_q(q)
    advantages = list(map(util.calc_advantage, trajectory))
    rewards = list(map(lambda item: item.r, trajectory))
    seq = chainl(
        trajectory,
        # State is current frame plus N previous and N next frames
        (map, lambda item: item.s),
        # Get the middle channel (current image in video) from state
        (map, lambda state: state[int(len(state) / 2)]),
        list,
    )
    window[EVALUATION_CANVAS].redraw_plot(advantages, rewards)
    window[VIDEO].set_video(seq, window, start_animation=True)


## Event dispatchers
@dispatch_on(VIDEO_SELECTOR)
def _(values, window):
    (selected_video,) = values[VIDEO_SELECTOR]
    handle_video_file_selected(selected_video, window)


@dispatch_on(VIDEO_SELECTOR_FIRST_BUTTON)
def _(values, window):
    video_selector = window[VIDEO_SELECTOR]
    video_selector.update(set_to_index=0, scroll_to_index=0)
    (selected_video,) = video_selector.get()
    handle_video_file_selected(selected_video, window)


@dispatch_on(VIDEO_SELECTOR_PREVIOUS_BUTTON)
def _(values, window):
    video_selector = window[VIDEO_SELECTOR]
    (current_video_index,) = video_selector.get_indexes()
    previous_video_index = max(current_video_index - 1, 0)
    scroll_index = previous_video_index
    video_selector.update(
        set_to_index=previous_video_index, scroll_to_index=scroll_index
    )
    (selected_video,) = video_selector.get()
    handle_video_file_selected(selected_video, window)


@dispatch_on(VIDEO_SELECTOR_NEXT_BUTTON)
def _(values, window):
    video_selector = window[VIDEO_SELECTOR]
    (current_video_index,) = video_selector.get_indexes()
    next_video_index = min(
        current_video_index + 1, len(video_selector_file_options) - 1
    )
    scroll_index = next_video_index
    video_selector.update(set_to_index=next_video_index, scroll_to_index=scroll_index)
    (selected_video,) = video_selector.get()
    handle_video_file_selected(selected_video, window)


@dispatch_on(VIDEO_SELECTOR_LAST_BUTTON)
def _(values, window):
    video_selector = window[VIDEO_SELECTOR]
    last_video_index = len(video_selector_file_options) - 1
    video_selector.update(
        set_to_index=last_video_index, scroll_to_index=last_video_index
    )
    (selected_video,) = video_selector.get()
    handle_video_file_selected(selected_video, window)


@dispatch_on(VIDEO_SELECTOR_SORT_BY)
def _(values, window):
    # Before re-sorting (and thus resetting the dropbox values), let's get the currently selected value.
    (selected_video_file,) = values[VIDEO_SELECTOR]

    # Sort the video_selector_file_options based on the selected_sort_order.
    selected_sort_order = values[VIDEO_SELECTOR_SORT_BY].value
    video_selector_file_options.sort(
        key=video_selector_file_options_sort_fn(selected_sort_order)
    )

    # Now let's re-render the video_selector with the re-sorted video_selector_file_options.
    video_selector = window[VIDEO_SELECTOR]
    # Reset the dropbox options.
    video_selector.update(values=video_selector_file_options)
    # Re-select the selected value.
    video_selector.set_value([selected_video_file])
    # Scroll to the selected value.
    (current_index,) = video_selector.get_indexes()
    video_selector.update(scroll_to_index=current_index)


## Main loop


def start_event_loop(window):
    # Initialize stuff...
    window.read(timeout=10)
    handle_video_file_selected(video_selector_file_options[0], window)

    while True:
        # To make keyboard events work with this setup, check out https://www.gitmemory.com/issue/PySimpleGUI/PySimpleGUI/64/757099987
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        else:
            handle_event(event, values, window)


window = sg.Window(
    "Performance insight",
    layout.get_layout(sort_options, video_selector_file_options),
)
start_event_loop(window)
window.close()
