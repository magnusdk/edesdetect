from collections import defaultdict

event_handlers = defaultdict(list)


def dispatch_on(key):
    """Decorator function for adding event dispatchers.

    Define dispatchers as:
    @dispatch_on(VIDEO_SELECTOR)
    def video_selector_dispatcher(values, window):
        print("Video selector event just happened :)")
    """

    def inner(func):
        event_handlers[key].append(func)
        return func

    return inner


def handle_event(event, values, window):
    for event_handler in event_handlers.get(event):
        event_handler(values, window)
