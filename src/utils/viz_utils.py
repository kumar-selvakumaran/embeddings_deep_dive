from IPython.display import Image as im 
from IPython.display import display as dis 
import matplotlib.pyplot as plt
import glob 
import cv2

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

    def viz_plot_data(self, title="Image Grid",
                      mode = "square",
                      subplot_title_hspace = 0.23,
                      subplot_title_wspace = 0.1):
        
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
                
            ax.axis('off')
            ax.set_title(self.titles[i], fontsize= 9)
            
            if hasattr(img, 'figure'):
                for ax_child in img.axes:
                    for line in ax_child.get_lines():
                        ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())
            else:
                ax.imshow(img[:, :, ::-1])

        total_num_plots = (total_rows * square_size)
        for i in range(total_num_plots - (total_num_plots - num_images), total_num_plots):
            axs.flat[i].set_visible(False)
            

        plt.axis('off')
        plt.tight_layout()
        # Adjust the spacing
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=subplot_title_wspace,
                            hspace=subplot_title_hspace)
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


def get_masked_crop(embedding_details, object_ind, masks_available = False, show_full_img = False):
    source_path = embedding_details['source_paths'][object_ind]
    image = cv2.imread(source_path)
    bbox = embedding_details["bounding_boxes"][object_ind][0]
    class_name =  embedding_details["class_names"][object_ind]
    # print(f"class name : {class_name}")
    # print(f"source path : {source_path}")
    # print(f"object_ind : {object_ind}")
    
    if masks_available:
        mask = embedding_details["masks"][object_ind][0]
        image = image * np.concatenate([mask[..., None],
                                            mask[..., None],
                                            mask[..., None]], axis = 2)
    
    if not show_full_img:
        ymin, xmin, ymax, xmax = bbox.astype(int)
        image = image[xmin:xmax, ymin:ymax, :]

    else:
        ymin, xmin, ymax, xmax = bbox.astype(int)
        vizimg = np.zeros_like(image)
        vizimg[xmin:xmax, ymin:ymax, :] = image[xmin:xmax, ymin:ymax, :]
        image = vizimg

    return image

def plot_neighbours(embedding_details,
                    neighbour_inds,
                    masks_available = False,
                    title = "enter model used as title",
                    small_titles = False,
                    show_full_img = False,
                    subplot_title_hspace = 0.23,
                    subplot_title_wspace = 0.1):
    
    main_title = title

    plotter = concise_ims_and_plots()

    num_embeddings, num_neighbours = neighbour_inds.shape
    for object_ind in range(num_embeddings):
        masked_image = get_masked_crop(embedding_details, object_ind, masks_available = masks_available, show_full_img = show_full_img)
        class_name = embedding_details["class_names"][object_ind]
        title =  f"T:{class_name}" if not small_titles else "T"
        plotter.add_plot_data(masked_image, title)
        for i, neighbour_ind in enumerate(neighbour_inds[object_ind]):
            masked_image = get_masked_crop(embedding_details, neighbour_ind, masks_available = masks_available, show_full_img = show_full_img)
            class_name = embedding_details["class_names"][neighbour_ind]
            title =  f"M:{num_neighbours - i}: {class_name}" if not small_titles else "M"
            plotter.add_plot_data(masked_image, title)

    plotter.viz_plot_data(title = main_title,
                          subplot_title_hspace = subplot_title_hspace,
                          subplot_title_wspace = subplot_title_wspace)


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


def apply_pca(data, n_components=2):
    """ Apply PCA to reduce dimensions to n_components. """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data), pca

def apply_tsne(data, n_components=2, perplexity=30):
    """ Apply t-SNE to reduce dimensions to n_components. """
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    return tsne.fit_transform(data), tsne

def plot_embeddings(embeddings, title, colors, xscale = None, yscale = None):
    """ Plot the 2D embeddings with matplotlib. """

    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True)
    
    if xscale != None:
        ax.set_xlim(xscale[0], xscale[1])
    
    if yscale != None:
        ax.set_ylim(yscale[0], yscale[1])
        
    plt.close(fig)  # Prevent it from showing immediately
    return fig