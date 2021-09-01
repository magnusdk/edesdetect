import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class QPlot(sg.Canvas):
    def __init__(self, *args, **kwargs):
        self.is_initialized = False
        super().__init__(*args, **kwargs)

    def redraw_plot(self, advantage, rewards):
        if not self.is_initialized:
            fig, ax = plt.subplots()
            self.fig = fig
            self.ax = ax
            self.canvas_agg = FigureCanvasTkAgg(self.fig, self.TKCanvas)
            self.is_initialized = True

        self.ax.cla()
        self.ax.plot(advantage)
        self.ax.plot(rewards)
        self.ax.legend(["Diastole", "Systole", "Reward"])
        self.canvas_agg.draw()
        self.canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
