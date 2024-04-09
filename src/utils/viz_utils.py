from IPython.display import Image as im 
from IPython.display import display as dis 
import matplotlib.pyplot as plt
import glob 
import cv2

import numpy as np
import math

# Function from previous instruction

class concise_ims_and_plots:
    """
    easily plot images and/or plots together in a square subgrid

    usage : 

    
    def generate_plot():
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        plt.close(fig)  # Prevent it from showing immediately
        return fig

    plotter = concise_ims_and_plots()

    plotting_data = []

    for i in range(10):
        plot = generate_plot()
        image = np.random.rand(10, 10, 3)

        plotter.add_plot_data(plot, f'plot {i}')
        plotter.add_plot_data(image, f'image {i}')
    """
    def __init__(self,
                xdim = 15,
                ydim = 15
                ):
        self.plot_data = []
        self.titles = []
        self.xdim = xdim
        self.ydim = ydim
    
    def add_plot_data(self, plot, title):
        self.plot_data.append(tuple([plot]))
        self.titles.append(title)

    def clear_plot_data(self):
        self.plot_data = []

    def viz_plot_data(self, title="Image Grid", mode = "square"):
        num_images = len(self.plot_data)
        grid_size = math.ceil(math.sqrt(num_images))
            
        square_size = int(np.floor(np.sqrt(num_images)))
        total_plots_in_square = square_size ** 2
        additional_plots = num_images - total_plots_in_square
        additional_rows = int(np.ceil(additional_plots / square_size))

        total_rows = square_size + additional_rows

        
        if mode == "square":
            fig, axs = plt.subplots(total_rows, square_size, figsize=(self.xdim, self.ydim))
        elif mode == "column":
            fig, axs = plt.subplots(num_images, 1, figsize=(self.xdim, self.ydim))
        elif mode == "row":
            fig, axs = plt.subplots(1, num_images, figsize=(self.xdim, self.ydim))

        fig.suptitle(title, fontsize = 30)
        

        for i in range(num_images):
            axs.flat[i].axis("off")
        
        for i, image in enumerate(self.plot_data):
            img = image[0]

            if num_images == 1:
                ax = axs
            else:
                ax = axs.flat[i]
                
            ax.axis('on')
            ax.set_title(self.titles[i])
            
            if hasattr(img, 'figure'):
                for ax_child in img.axes:
                    for line in ax_child.get_lines():
                        ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())
            else:
                ax.imshow(img[:, :, ::-1])
        plt.tight_layout()
        plt.show()

def get_bar_plot(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    plt.close(fig)  # Prevent it from showing immediately
    return fig

def viz_im_small(impath):
    vizim = cv2.imread(impath)
    cv2.resize(vizim, (100,100))
    plt.imshow(vizim[:, :, ::-1])
    plt.show()


def im_in_window(image,
                 window_title = "image visualization"):
    viz_window = cv2.namedWindow(window_title)
    cv2.imshow(viz_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()