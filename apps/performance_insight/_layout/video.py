import threading

import numpy as np
import PySimpleGUI as sg
from PIL import Image, ImageTk


class AnimationTimer:
    def __init__(self, interval) -> None:
        self.interval = interval
        self._timer = None

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, callback):
        self._callback = callback

    def start(self):
        self.stop()
        self._timer = threading.Timer(self.interval, self._call)
        self._timer.start()

    def stop(self):
        if self._timer is not None:
            self._timer.cancel()

    def _call(self):
        self._callback()
        self.start()


class Video(sg.Image):
    """Subclass for animating a PySimpleGUI.Image from a numpy array.

    Construct this the same way as you would that of a PySimpleGUI.Image. Additional functionality is added as additional object methods.
    Call set_video(video, window) to render the first frame of the video. It is important that TKinter has been initialized before this step.
    Call start_animation(window) to start the animation, and likewise stop_animation() to stop it.

    video, as set by set_video(video, window), should have the shape (num_frames, width, height).
    Every frame in video is converted to a PIL.ImageTk.PhotoImage (and cached until next time set_video(video, window) is called).
    """

    def __init__(self, *args, **kwargs):
        self._timer = AnimationTimer(0.1)
        super().__init__(*args, **kwargs)

    def _get_current_imagetk(self):
        if self.imagetk_cache[self.frame] is not None:
            return self.imagetk_cache[self.frame]
        else:
            image = Image.fromarray(np.uint8(self.video[self.frame]))
            if self.Size != (None, None):
                image = image.resize(self.Size)
            imagetk = ImageTk.PhotoImage(image)
            self.imagetk_cache[self.frame] = imagetk
            return imagetk

    def set_video(self, video, window, start_animation=False):
        self.video = video
        self.num_frames = video.shape[0]
        self.imagetk_cache = np.repeat(None, self.num_frames)
        self.frame = 0

        imagetk = self._get_current_imagetk()
        super().update(data=imagetk)
        window.refresh()

        if start_animation:
            self.start_animation(window)

    def next_frame(self, window):
        self.frame = (self.frame + 1) % self.num_frames
        imagetk = self._get_current_imagetk()
        super().update(data=imagetk)
        window.refresh()

    def start_animation(self, window):
        self._timer.callback = lambda: self.next_frame(window)
        self._timer.start()

    def stop_animation(self):
        self._timer.stop()
