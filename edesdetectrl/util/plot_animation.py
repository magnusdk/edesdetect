from matplotlib.animation import FuncAnimation


def plot_animation(data, fig, ax):
    im = ax.imshow(data[0])

    def animate(i):
        im.set_data(data[i])

    anim = FuncAnimation(
        fig, animate, frames=data.shape[0], interval=50
    )

    return im, anim
