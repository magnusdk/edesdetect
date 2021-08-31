import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class QPlot(sg.Canvas):
    def __init__(self, *args, **kwargs):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.canvas_agg = None
        super().__init__(*args, **kwargs)

    def redraw_plot(self, advantage, rewards):
        self.ax.cla()
        self.ax.plot(advantage)
        self.ax.plot(rewards)
        self.ax.legend(["Diastole", "Systole", "Reward"])

        if self.canvas_agg is None:
            self.canvas_agg = FigureCanvasTkAgg(self.fig, self.TKCanvas)
        self.canvas_agg.draw()
        self.canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
