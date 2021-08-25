import PySimpleGUI as sg

VIDEO_SELECTOR = "VIDEO_SELECTOR"
VIDEO_SELECTOR_SORT_BY = (VIDEO_SELECTOR, "SORT_BY")
VIDEO_SELECTOR_FIRST_BUTTON = (VIDEO_SELECTOR, "FIRST_BUTTON")
VIDEO_SELECTOR_PREVIOUS_BUTTON = (VIDEO_SELECTOR, "PREVIOUS_BUTTON")
VIDEO_SELECTOR_NEXT_BUTTON = (VIDEO_SELECTOR, "NEXT_BUTTON")
VIDEO_SELECTOR_LAST_BUTTON = (VIDEO_SELECTOR, "LAST_BUTTON")


def get_layout(sort_options, video_selector_files_options):
    graph_timeline_layout = [
        [sg.Text("There will be a graph here")],
        [sg.Text("There will be a timeline here")],
    ]

    video_layout = [[sg.Text("There will be a video playing here")]]

    video_selector = [
        [
            sg.Text("Sort by"),
            sg.DropDown(
                sort_options,
                key=VIDEO_SELECTOR_SORT_BY,
                default_value=sort_options[0],
                readonly=True,
                enable_events=True,
            ),
            sg.Button("First", key=VIDEO_SELECTOR_FIRST_BUTTON),
            sg.Button("Previous", key=VIDEO_SELECTOR_PREVIOUS_BUTTON),
            sg.Button("Next", key=VIDEO_SELECTOR_NEXT_BUTTON),
            sg.Button("Last", key=VIDEO_SELECTOR_LAST_BUTTON),
        ],
        [
            sg.Listbox(
                video_selector_files_options,
                key=VIDEO_SELECTOR,
                select_mode=sg.SELECT_MODE_SINGLE,
                font=("Courier", sg.DEFAULT_FONT[1]),
                default_values=video_selector_files_options[0],
                expand_x=True,
                size=(None, 5),
                enable_events=True,
            )
        ],
        [
            sg.Text(
                "There will be informational text here about loading of files, etc...",
                font=(sg.DEFAULT_FONT[0], sg.DEFAULT_FONT[1], "italic"),
            )
        ],
    ]

    layout = [
        [sg.Column(graph_timeline_layout), sg.VSeperator(), sg.Column(video_layout)],
        [sg.HSeparator()],
        [video_selector],
    ]

    return layout
