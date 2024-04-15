from IPython.display import Image as im 
from IPython.display import display as dis 
import matplotlib.pyplot as plt
import glob 
import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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


def get_masked_crop(embedding_details, object_ind, masks_available = False):
    source_path = embedding_details['source_paths'][object_ind]
    image = cv2.imread(source_path)
    bbox = embedding_details["bounding_boxes"][object_ind][0]
    class_name =  embedding_details["class_names"][object_ind]
    print(f"class name : {class_name}")
    print(f"source path : {source_path}")
    print(f"object_ind : {object_ind}")
    
    if masks_available:
        mask = embedding_details["masks"][object_ind][0]
        image = image * np.concatenate([mask[..., None],
                                            mask[..., None],
                                            mask[..., None]], axis = 2)
    

    ymin, xmin, ymax, xmax = bbox.astype(int)
    image = image[xmin:xmax, ymin:ymax, :]

    return image

def plot_neighbours(embedding_details, neighbour_inds, masks_available = False):
        
    plotter = concise_ims_and_plots()

    num_embeddings, num_neighbours = neighbour_inds.shape
    for object_ind in range(num_embeddings):
        masked_image = get_masked_crop(embedding_details, object_ind, masks_available = masks_available)
        class_name = embedding_details["class_names"][object_ind]
        plotter.add_plot_data(masked_image, f"TARGET : {class_name}")
        for i, neighbour_ind in enumerate(neighbour_inds[object_ind]):
            masked_image = get_masked_crop(embedding_details, neighbour_ind, masks_available = masks_available)
            class_name = embedding_details["class_names"][neighbour_ind]
            plotter.add_plot_data(masked_image, f"match {num_neighbours - i}: {class_name}")

    plotter.viz_plot_data()


def initialize_maximally_spaced_colors(n_colors):
    colors_hsv = [(i * 360 / n_colors, 1, 1) for i in range(n_colors)]
    colors_rgb = [tuple(int(c * 255) for c in cv2.cvtColor(np.array([[hsv_color]], dtype=np.float32), cv2.COLOR_HSV2BGR)[0,0]) for hsv_color in colors_hsv]
    return colors_rgb

def draw_bounding_boxes(image, bounding_boxes, class_names):
    n_classes = len(class_names)
    colors = initialize_maximally_spaced_colors(n_classes)
    thickness = max(1, int((image.shape[0] + image.shape[1]) / 1000))
    font_scale = max(0.5, image.shape[0] / 1000)

    for box in bounding_boxes:
        xmin, ymin, xmax, ymax, confidence, class_idx = box
        class_idx = int(class_idx)
        label = f"{class_names[class_idx]}: {confidence:.2f}"
        color = colors[class_idx]

        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (int(xmin), int(ymin - label_height - baseline)), (int(xmin + label_width), int(ymin)), color, cv2.FILLED)
        cv2.putText(image, label, (int(xmin), int(ymin - baseline)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

    return image


def create_frame(images, titles, figsize=(10, 5)):
    layout = (1, len(images))  # Change layout to have enough spots for each image
    
    total_subplots = layout[0] * layout[1]

    fig, axs = plt.subplots(layout[0], layout[1], figsize=figsize)
    axs = axs.flatten() if layout[0] * layout[1] > 1 else [axs]
    
    # Hide any unused subplots
    for ax in axs[len(images):]:
        ax.axis('off')
    
    for ax, img, title in zip(axs[:len(images)], images, titles):
        ax.imshow(img, aspect='auto')
        ax.set_title(title)
        ax.axis('off')
    
    # plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust spacing to prevent overlap
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    frame = np.array(canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return frame
