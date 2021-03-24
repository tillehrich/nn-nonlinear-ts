import matplotlib.pyplot as plt
import numpy as np

def logistic(x):
    return 1/(1+np.exp(-x))

def tanh(x):
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
    return max(0,x)

relu = np.vectorize(relu)

#adapted from https://gist.github.com/anbrjohn/7116fa0b59248375cd0c0371d6107a59
def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_text=None):
    '''
    Draw a neural network cartoon using matplotilb.

    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2], ['x1', 'x2','x3','x4'])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
        - layer_text : list of str
            List of node annotations in top-down left-right order
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    ax.axis('off')
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            x = n * h_spacing + left
            y = layer_top - m * v_spacing
            circle = plt.Circle((x, y), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Node annotations
            if layer_text:
                text = layer_text.pop(0)
                plt.annotate(text, xy=(x, y), zorder=5, ha='center', va='center', fontsize = 15)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)



if __name__ == '__main__':
    x = np.linspace(-6,6, 1000).astype(int)
    y_log = logistic(x)
    y_tanh = tanh(x)
    y_relu = relu(x)

    #plot activation functions
    plt.plot(x,y_log, label = 'logistic')
    plt.plot(x,y_tanh,label = 'tanh')
    plt.plot(x, y_relu, label='relu')
    plt.ylim((-1.5, 1.5))
    plt.legend()
    plt.savefig('../../manuscript/src/latexGraphics/actfunctions.png')
    plt.show()
    plt.close()

    #plot neural network structure
    #text for nodes
    node_text = ['$y_{t-1}$', '$y_{t-2}$', '$y_{t-3}$', '$y_{t-4}$', '$Z_1$', '$Z_2$', '$Z_3$', '$Z_4$', '$y_t$']

    #dictionary of annotated weights and their position
    weight_dct = {r'$\alpha_{11}$': (0.175, 0.815),
                  r'$\alpha_{21}$': (0.195, 0.765),
                  r'$\alpha_{31}$': (0.184, 0.72),
                  r'$\alpha_{41}$': (0.115, 0.71),
                  r'$\alpha_{12}$': (0.14, 0.65),
                  r'$\beta_1$': (0.7, 0.67),
                  r'$\beta_2$': (0.7, 0.565),
                  r'$\beta_3$': (0.7, 0.47),
                  r'$\beta_4$': (0.7, 0.38)}

    fontsize_weights = 15

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    draw_neural_net(ax, .1, .9, .1, .9, [4, 4, 1], node_text)

    for weight in weight_dct:
        plt.text(weight_dct[weight][0], weight_dct[weight][1], weight, fontsize=fontsize_weights)
    plt.savefig('../../manuscript/src/latexGraphics/nnstructure.png')
    plt.show()