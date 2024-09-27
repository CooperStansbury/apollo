import sys
from tifffile import imread
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation


def make_animation(img,output_path):
    """
    Creates and saves a matplotlib animation from a sequence of images (img).

    Args:
        img: A list or array of image data (e.g., numpy arrays).
        output_path: The file path to save the animation GIF.
    """

    fig, ax = plt.subplots()

    # Define the colors and their positions
    colors = [
        (0.0, 'white'), 
        (0.33, 'red'), 
        (0.66, 'green'), 
        (1.0, 'yellow'),
    ]

    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('white_red_green_yellow', colors, N=256)

    # Display the first image
    im = ax.imshow(img[0], cmap=cmap, interpolation='nearest')

    plt.yticks([])
    plt.xticks([])

    # Animation function
    def animate(i):
        im.set_array(img[i])
        return im,

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        animate, 
        frames=len(img), 
        interval=250, 
        blit=True,
    )

    # Save the animation
    ani.save(output_path, writer='pillow')
    
    
if __name__ == "__main__":
    img_path = sys.argv[1]
    out_path = sys.argv[2]
    
    img = imread(img_path)
    make_animation(img, out_path)
  
    
