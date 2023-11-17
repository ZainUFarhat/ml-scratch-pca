# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# visualize pca
def visualize_pca(X_projected, y, dataset_name, class_names, savepath, is_iris):

    """
    Description:
        Visualizes our projected data after performing PCA
    
    Parameters:
        X_projected: our projected data
        y: labels, used to color our plots
        dataset_name: name of our dataset
        class_names: our given labels, but PCA doesn't need labels, used only for plotting purposes
        savepath: where to save our plots to
        is_iris: boolean variable if our dataset is iris, this is for plotting the cmap for iris (unecessary I know!)

    Returns:
        None
    """

    # extract our components to plot
    x1, x2 = X_projected[:, 0], X_projected[:, 1]

    plt.figure(figsize = (10, 10))

    # set background color to lavender
    ax = plt.axes()
    ax.set_facecolor("lavender")

    if is_iris:
        sc = plt.scatter(x1, x2, c = y, cmap = plt.cm.get_cmap('viridis', 3))
        handles, _ = sc.legend_elements()
        plt.legend(handles, class_names, title = 'Classes')
    else:
        colors = ['r', 'b']
        sc = plt.scatter(x1, x2, c = y, label = class_names, cmap = ListedColormap(colors))
        handles, _ = sc.legend_elements()
        plt.legend(handles, class_names, title = 'Classes')
    
    plt.title(f'{dataset_name} - Principal Component Analysis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()

    plt.savefig(savepath)