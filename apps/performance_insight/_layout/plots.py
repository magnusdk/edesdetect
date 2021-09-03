import matplotlib.pyplot as plt
import PySimpleGUI as sg
from apps.performance_insight.util import throttle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class QPlot(sg.Canvas):
    def __init__(self, mouseover_callback, *args, **kwargs):
        self.mouseover_callback = mouseover_callback
        self.is_initialized = False
        super().__init__(*args, **kwargs)

    def redraw_xline(self):
        if self.xline is None:
            self.xline = self.ax.axvline(self.selected_frame)
        else:
            self.xline.set_xdata(self.selected_frame)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    @throttle(0.1)
    def on_motion(self, event):
        if event.xdata is not None:
            self.selected_frame = int(event.xdata + 0.5)
            self.redraw_xline()
            self.mouseover_callback(self.selected_frame)

    def redraw_plot(self, advantage, rewards):
        if not self.is_initialized:
            fig, ax = plt.subplots()
            self.fig = fig
            self.ax = ax
            self.canvas_agg = FigureCanvasTkAgg(self.fig, self.TKCanvas)
            self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
            self.is_initialized = True

        self.xline = None
        self.ax.cla()
        self.ax.plot(advantage)
        self.ax.plot(rewards)
        self.ax.legend(["Diastole", "Systole", "Reward"])
        self.canvas_agg.draw()
        self.canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
