import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

def improved_coloring(journals, dict_words_colors):
    """Creates coloring based on words appearing in a list of documents.
    It creates an array with colors, assigning a color to each paper depending on whether it contains a word in its journal title from the keys in ` dict_words_colors`.

    IMPORTANT REMARK: if the journal name contains two words belonging to the word list, the color of the word
    located the latest in the list will be assigned to it (first, the first word's color is assigned and then
    the second overwrites the first).

    Parameters
    ----------
    journals : dataframe of str
        Dataframe with the journal names of the papers, or any other corpus where to look for the words.
    dict_words_colors : dict
        Dictionary matching words to colors (legend). The keys are the words and the values are the colors.


    Returns
    -------
    labels_with_unlabeled : list of str fo len (n_journals)
        List or labels (words) for all instances including label 'unlabeled'.
    colors : array
        Colors for each paper.

    See Also
    --------
    automatic_coloring

    """

    words = dict_words_colors.keys()
    labels = np.empty(len(journals))

    for i, wrd in enumerate(words):
        word_may = wrd.capitalize()
        word_min = " " + wrd

        indexes1 = journals.str.find(word_may)
        indexes2 = journals.str.find(word_min)

        labels = np.where((indexes1 != -1) | (indexes2 != -1), wrd, labels)

    # create colors
    colors = np.vectorize(dict_words_colors.get)(labels)

    # add grey to the rest of papers
    colors = np.where(colors == None, "lightgrey", colors)
    colors = np.where(colors == "None", "lightgrey", colors)

    # change 0 for 'unlabeled'
    labels_with_unlabeled = np.where(
        colors == "lightgrey", "unlabeled", labels
    )

    return labels_with_unlabeled, colors

def plot_tsne_colors(
    tsne,
    colors,
    x_lim=None,
    y_lim=None,
    ax=None,
    plot_type=None,
    axis_on=False,
):
    """Plot t-SNE embedding with colors (by labels).

    Parameters
    ----------
    tsne: array-like
        t-SNE coordinates.
    colors : array-like
        Color values for the colormap.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    plot_type : {None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    axis_on : bool, default=False
        If True, axis is shown in plot.

    """
    if x_lim is not None:
        assert x_lim[0] < x_lim[1], "xlim values are in the wrong order."
    if y_lim is not None:
        assert y_lim[0] < y_lim[1], "ylim values are in the wrong order."

    assert plot_type in [
        None,
        "subplot_2",
        "subplot_3",
        "subplot_3_grey",
        "subregion",
        "test",
        "pdf ML",
    ], "Not valid `plot_type` value. Choose from [None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test', 'pdf ML']."

    if ax is None:
        fig, ax = plt.subplots()

    s_grey = 5
    s_color = 5
    alpha_grey = 0.6
    alpha_color = 0.7

    # if plot_type == "subplot_2":
    #     s_grey = 0.2
    #     s_color = 0.2

    if plot_type == "subplot_3":
        s_grey = 3
        s_color = 3

    # if plot_type == "subplot_3_grey":
    #     s_grey = 0.01
    #     alpha_grey = 0.01
    #     s_color = 0.2
    #     alpha_color = 0.5

    # if plot_type == "subregion":
    #     s_grey = 1
    #     s_color = 1
    #     alpha_grey = 0.6
    #     alpha_color = 0.7

    # if plot_type == "test":
    #     s_grey = 2
    #     s_color = 2
    #     alpha_grey = 0.6
    #     alpha_color = 0.7

    # if plot_type == "pdf ML":
    #     s_grey = 0.5
    #     alpha_grey = 0.02

    #     # s_grey = 0.2
    #     # alpha_grey = 0.2
    #     s_color = 0.2
    #     alpha_color = 0.5

    ax.scatter(
        tsne[:, 0][colors == "lightgrey"],
        tsne[:, 1][colors == "lightgrey"],
        s=s_grey,
        alpha=alpha_grey,
        c="lightgrey",
        marker=".",
        linewidths=0,
        ec="None",
        rasterized=True,
    )
    ax.scatter(
        tsne[:, 0][colors != "lightgrey"],
        tsne[:, 1][colors != "lightgrey"],
        s=s_color,
        alpha=alpha_color,
        c=colors[colors != "lightgrey"],
        marker=".",
        linewidths=0,
        ec="None",
        rasterized=True,
    )

    if plot_type == "subregion":
        ax.axis("scaled")
    else:
        ax.axis("scaled")

    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    if axis_on == False:
        ax.axis("off")


def find_cluster_center(tsne, colors, legend, subset = True, subset_size = 500000, rs = 42):
    """Find cluster centers.
    Finds coordinates of the highest density point of points from each label, using gaussian_kde.
    
    Parameters
    ----------
    tsne: array-like of shape (n_points,2)
        t-SNE coordinates.
    colors : array-like of shape (n_points,)
        Color values for the colormap.
    legend : dict
        Legend label-color.
    subset : bool, default= True
         If True, a subset of the dataset is used for the cluster center calculations.
    subset_size : int, default=500000
        Size of the subset of the dataset used for the cluster center calculations.
    rs : int, default= 42
         Random seed.
         
    Returns
    -------
    center_cluster_coordinates_df : dataframe of shape (n_clusters, 2)
        Cluster center coordinates stored in two columns: "x" and "y".
    

    """
    
    words = list(legend.keys())
    unique_colors = np.array(list(legend.values()))
    
    if subset == True:
        np.random.seed(rs)
        assert tsne.shape[0] >= subset_size, "Subset size is smaller than dataset"
        index_subset=np.random.randint(0,tsne.shape[0],subset_size)
        tsne_subset=tsne[index_subset,:]
        colors_subset=colors[index_subset]
        
    else:
        tsne_subset=tsne
        colors_subset=colors
    
    # calculate cluster centers
    center_cluster_coordinates = []
    for i in range(len(words)):
        cluster=tsne_subset[colors_subset==unique_colors[i]]
        assert cluster.shape[0] > 0, print(words[i])
        #center with kernel density
        kde = gaussian_kde(cluster.T)
        center_cluster_coordinates.append(cluster[kde(cluster.T).argmax()])
    
    center_cluster_coordinates = np.vstack(center_cluster_coordinates)
    
    center_cluster_coordinates_df = pd.DataFrame(center_cluster_coordinates, index = words, columns = ['x', 'y'])
    
    return center_cluster_coordinates_df



def plot_label_tags(tsne, colors, legend, x_lim, y_lim, ax=None, middle_value = 0, subset = True, subset_size = 500000, rs = 42, fontsize=7, capitalize=True):
    """Plots label tags and a line pointing to the embedding.
    The line from a label tag points to the location with higher points density of that specific label.
    
    
    Parameters
    ----------
    tsne: array-like of shape (n_points,2)
        t-SNE coordinates.
    colors : array-like of shape (n_points,)
        Color values for the colormap.
    legend : dict
        Legend label-color.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created. 
    middle_value : float, default=0
         The x value to decide which labels go to the left and which go to the right.
    subset : bool, default= True
         If True, a subset of the dataset is used for the cluster center calculations.
    subset_size : int, default=500000
        Size of the subset of the dataset used for the cluster center calculations.
    rs : int, default= 42
         Random seed.
    fontsize: int, default=7
         Fontsize for the labels.
    capitalize : bool, default = True
        If True, it will capitalize the labels.
    
    See Also
    --------
    find_cluster_center
    
    """
    
    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"
    
    if ax is None:
        fig, ax = plt.subplots()

    if "unlabeled" in set(legend.keys()):
        legend.pop("unlabeled")

    # calculate cluster centers
    center_cluster_coordinates = find_cluster_center(tsne, colors, legend, subset, subset_size, rs)
    
    # sort by x
    center_cluster_coordinates_left = center_cluster_coordinates[center_cluster_coordinates.x < middle_value].copy()
    center_cluster_coordinates_right = center_cluster_coordinates[center_cluster_coordinates.x >= middle_value].copy()

    # sort by y
    center_cluster_coordinates_left.sort_values(by = 'y', inplace=True, ascending = False)
    center_cluster_coordinates_right.sort_values(by = 'y', inplace=True, ascending = False)
    
    sorted_labels_left = center_cluster_coordinates_left.index.tolist()
    sorted_labels_right = center_cluster_coordinates_right.index.tolist()
    if len(sorted_labels_left) != 0:
        sorted_colors_left = np.vectorize(legend.get)(sorted_labels_left)
    else:
        sorted_colors_left = []
        
    if len(sorted_labels_right) != 0:
        sorted_colors_right = np.vectorize(legend.get)(sorted_labels_right)
    else:
        sorted_colors_right = []
    
    if capitalize == True:
        sorted_labels_left = [elem.capitalize() for elem in sorted_labels_left]
        sorted_labels_right = [elem.capitalize() for elem in sorted_labels_right]


    # PLOT
    # left
    n_left=len(sorted_labels_left)
    x=x_lim[0]*np.ones(n_left)
    y=np.linspace(y_lim[1], y_lim[0], n_left)
    alpha_box=0.8

    for i, colr in enumerate(sorted_colors_left):
        if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba'] ):            
            # white colored letters
            ax.text(x[i], y[i], sorted_labels_left[i], c='lightgrey', fontsize=fontsize, ha='right', bbox=dict(facecolor=colr,edgecolor='None', alpha=alpha_box, boxstyle="Round, pad=0.075, rounding_size=0.3"))
            ax.plot([x[i],center_cluster_coordinates_left.x[i]],[y[i],center_cluster_coordinates_left.y[i]], c=colr, linewidth=0.4, clip_on=False)
        else:
            # black colored letters
            ax.text(x[i], y[i], sorted_labels_left[i], c='black', fontsize=fontsize, ha='right', bbox=dict(facecolor=colr,edgecolor='None', alpha=alpha_box, boxstyle="Round, pad=0.075, rounding_size=0.3"))
            ax.plot([x[i],center_cluster_coordinates_left.x[i]],[y[i],center_cluster_coordinates_left.y[i]], c=colr, linewidth=0.4, clip_on=False)

    # right
    n_right=len(sorted_labels_right)
    x=x_lim[1]*np.ones(n_right)
    y=np.linspace(y_lim[1], y_lim[0], n_right)

    for i, colr in enumerate(sorted_colors_right):
        # color blanco
        if any( [colr=='black', colr=='#0000A6', colr=='#5A0007', colr=='#4A3B53', colr=='#1B4400',
                 colr=='#004D43', colr=='#013349', colr=='#000035', colr=='#300018', colr=='#001E09',
                 colr=='#372101', colr=='#6508ba'] ):
            ax.text(x[i], y[i], sorted_labels_right[i], c='lightgrey', fontsize=fontsize, ha='left', bbox=dict(facecolor=colr,edgecolor='None', alpha=alpha_box, boxstyle="Round, pad=0.075, rounding_size=0.3"))
            ax.plot([x[i],center_cluster_coordinates_right.x[i]],[y[i],center_cluster_coordinates_right.y[i]], c=colr, linewidth=0.4, clip_on=False)
        else:
            ax.text(x[i], y[i], sorted_labels_right[i], c='black', fontsize=fontsize, ha='left', bbox=dict(facecolor=colr,edgecolor='None', alpha=alpha_box, boxstyle="Round, pad=0.075, rounding_size=0.3"))
            ax.plot([x[i],center_cluster_coordinates_right.x[i]],[y[i],center_cluster_coordinates_right.y[i]], c=colr, linewidth=0.4, clip_on=False)