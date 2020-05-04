import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def get_mesh(history, input_data, target, bounds=None):
    if bounds == None:
        x = np.linspace(np.floor(history[:, 0].min())-1., np.ceil(history[:, 0].max()+1.), 100)
        y = np.linspace(np.floor(history[:, 1].min())-1., np.ceil(history[:, 1].max()+1.), 100)
    else:
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[2], bounds[3], 100)
    X, Y = np.meshgrid(x, y)
    
    X_, Y_ = np.reshape(X, [-1]), np.reshape(Y, [-1])
    weights = np.stack([X_, Y_], -1)[..., None]
    prediction = np.einsum('bi,kij->kbj', input_data, weights)
    loss = np.mean((prediction - target[None, ...]) ** 2, 1)
    return X, Y, loss.reshape(100, 100)    


def plot_3d(history, input_data, target, bounds=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y, Z = get_mesh(history, input_data, target, bounds)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.2)

    ax.plot(history[:, 0], history[:, 1], history[:, 2])
    ax.view_init(elev=30., azim=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig


def plot_2d(history, input_data, target, bounds=None):
    X, Y, Z = get_mesh(history, input_data, target, bounds)
    norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
    plt.contourf(X, Y, Z, 20, norm=norm)
    plt.plot(history[:, 0], history[:, 1], color='orange')
    