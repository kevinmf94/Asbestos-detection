import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

data = pd.read_csv("experiment32/test_results.csv")

fig = plt.figure(figsize=(9., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 9),
                 axes_pad=0.1
                 )

for ax, item in zip(grid, data.loc[36*6:36*6+36, ('Image_crop', 'Class', 'Score')].to_numpy()):
    # Iterating over the grid returns the Axes.
    ax.imshow(Image.open(item[0]))
    ax.text(10, 50, f"{item[2]:.2f}", color="white", fontweight="bold", bbox=dict(fill=True, linewidth=0, color="orange"))
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])

    if item[1] == 1:
        ax.patch.set_edgecolor('green')
        ax.patch.set_linewidth('4')
    else:
        ax.patch.set_edgecolor('red')
        ax.patch.set_linewidth('4')

plt.show()