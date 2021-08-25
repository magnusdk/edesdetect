import random

import apps.performance_insight.layout as layout
import pandas as pd
import PySimpleGUI as sg
from apps.performance_insight.events import dispatch_on, handle_event
from apps.performance_insight.layout import (
    VIDEO_SELECTOR,
    VIDEO_SELECTOR_FIRST_BUTTON,
    VIDEO_SELECTOR_LAST_BUTTON,
    VIDEO_SELECTOR_NEXT_BUTTON,
    VIDEO_SELECTOR_PREVIOUS_BUTTON,
    VIDEO_SELECTOR_SORT_BY,
)
from edesdetectrl.config import config

SORT_OPTION_FILENAME = "filename"
SORT_OPTION_BEST = "best"
SORT_OPTION_WORST = "worst"


def present_video_selector_file_options(options):
    return [f"{option['filename']} ({option['perf']:.2f})" for option in options]


def video_selector_file_options_sort_fn(sort_order):
    if sort_order == SORT_OPTION_FILENAME:
        return lambda video: video["filename"]
    elif sort_order == SORT_OPTION_BEST:
        # Sort by negative perf, because lower perf should be sorted last
        return lambda video: -video["perf"]
    elif sort_order == SORT_OPTION_WORST:
        return lambda video: video["perf"]


class DropDownOption:
    def __init__(self, value, label):
        self.value = value
        self.label = label

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return str((self.value, self.label))


sort_options = [
    DropDownOption(SORT_OPTION_BEST, "Best"),
    DropDownOption(SORT_OPTION_WORST, "Worst"),
    DropDownOption(SORT_OPTION_FILENAME, "Filename"),
]

filelist_df = pd.read_csv(config["data"]["filelist_path"])
filenames = filelist_df["FileName"].tolist()
video_selector_file_options = [
    {"filename": filename, "perf": random.random()} for filename in filenames
]
video_selector_file_options.sort(
    key=video_selector_file_options_sort_fn(sort_options[0].value)
)


## Event dispatchers


@dispatch_on(VIDEO_SELECTOR)
def _(values, window):
    pass


@dispatch_on(VIDEO_SELECTOR_FIRST_BUTTON)
def _(values, window):
    window[VIDEO_SELECTOR].update(set_to_index=0, scroll_to_index=0)


@dispatch_on(VIDEO_SELECTOR_PREVIOUS_BUTTON)
def _(values, window):
    video_selector = window[VIDEO_SELECTOR]
    (current_video,) = video_selector.get_indexes()
    previous_video = max(current_video - 1, 0)
    scroll_index = previous_video
    video_selector.update(set_to_index=previous_video, scroll_to_index=scroll_index)


@dispatch_on(VIDEO_SELECTOR_NEXT_BUTTON)
def _(values, window):
    video_selector = window[VIDEO_SELECTOR]
    (current_video,) = video_selector.get_indexes()
    next_video = min(current_video + 1, len(video_selector_file_options) - 1)
    scroll_index = next_video
    video_selector.update(set_to_index=next_video, scroll_to_index=scroll_index)


@dispatch_on(VIDEO_SELECTOR_LAST_BUTTON)
def _(values, window):
    last_video = len(video_selector_file_options) - 1
    window[VIDEO_SELECTOR].update(set_to_index=last_video, scroll_to_index=last_video)


@dispatch_on(VIDEO_SELECTOR_SORT_BY)
def _(values, window):
    # Before re-sorting (and thus resetting the dropbox values), let's get the currently selected value.
    (selected_filename,) = values[VIDEO_SELECTOR]

    # Sort the video_selector_file_options based on the selected_sort_order.
    selected_sort_order = values[VIDEO_SELECTOR_SORT_BY].value
    video_selector_file_options.sort(
        key=video_selector_file_options_sort_fn(selected_sort_order)
    )

    # Now let's re-render the video_selector with the re-sorted video_selector_file_options.
    video_selector = window[VIDEO_SELECTOR]
    # Reset the dropbox options.
    video_selector.update(
        values=present_video_selector_file_options(video_selector_file_options)
    )
    # Re-select the selected value.
    video_selector.set_value(selected_filename)
    # Scroll to the selected value.
    (current_index,) = video_selector.get_indexes()
    video_selector.update(scroll_to_index=current_index)


## Main loop


def start_event_loop(window):
    while True:
        # To make keyboard events work with this setup, check out https://www.gitmemory.com/issue/PySimpleGUI/PySimpleGUI/64/757099987
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        else:
            handle_event(event, values, window)


window = sg.Window(
    "Performance insight",
    layout.get_layout(
        sort_options, present_video_selector_file_options(video_selector_file_options)
    ),
)
start_event_loop(window)
window.close()
