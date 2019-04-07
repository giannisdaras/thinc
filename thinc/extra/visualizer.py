''' A visualizer module for Thinc '''
import seaborn
import plt

def visualize_attention(x, y, weights):
    '''
        Visualize self/outer attention
        Args:
            x: sentence
            y: sentence
            weights:
                (nL, nL) or (nS, nL, nL)
    '''
    def heatmap(x, y, data, ax):
        seaborn.heatmap(data, xticklabels=x, yticklabels=y, vmin=0.0, vmax=1.0,
                        ax=ax)

    fix, axs = plt.subplots(1, weights.shape[0])
    if len(weights.shape) == 2:
        heatmap(x, y, weights, axs[0])
    elif len(weights.shape) == 3:
        for i in range(weights.shape[0]):
            heatmap(x, y, weights[i], axs[i])
    else:
        raise ValueError("Wrong input weights dimensions")
    plt.show()
