import threading

import numpy as np
import PySimpleGUI as sg
from PIL import Image, ImageTk


class Video(sg.Image):
    """Subclass for animating a PySimpleGUI.Image from a numpy array.
    
    Construct this the same way as you would that of a PySimpleGUI.Image. Additional functionality is added as additional object methods.
    Call set_video(video, window) to render the first frame of the video. It is important that TKinter has been initialized before this step.
    Call start_animation(window) to start the animation, and likewise stop_animation() to stop it.

    video, as set by set_video(video, window), should have the shape (num_frames, width, height).
    Every frame in video is converted to a PIL.ImageTk.PhotoImage (and cached until next time set_video(video, window) is called).
    """
    def __init__(self, *args, **kwargs):
        self.FPS = 0.1
        super().__init__(*args, **kwargs)

    def _get_current_imagetk(self):
        if self.imagetk_cache[self.frame] is not None:
            return self.imagetk_cache[self.frame]
        else:
            im_data = np.uint8(self.video[self.frame])
            imagetk = ImageTk.PhotoImage(Image.fromarray(im_data))
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

    def _animate(self, window):
        self.next_frame(window)
        if self.animating:
            threading.Timer(self.FPS, lambda: self._animate(window)).start()

    def start_animation(self, window):
        self.animating = True
        threading.Timer(self.FPS, lambda: self._animate(window)).start()

    def stop_animation(self):
        self.animating = False
