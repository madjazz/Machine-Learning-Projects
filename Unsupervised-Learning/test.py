def plot_dendrograms(labels, models):
    fig = plt.figure()
    fig.suptitle("Dendrograms ", fontsize=16)
    for i in range(1, len(labels)):
        ax = plt.subplot(7, 2, i)
        ax.set_title(labels[i-1])
        dendrogram(models[i-1])
    plt.show()
