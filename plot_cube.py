import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_cube(cube_size, orig_point=None):
    if orig_point is None:
        orig_point = [0, 0, 0]
    point_all_unit_cube = np.asarray(
        [[0, 0, 0],
         [1, 0, 0],
         [1, 1, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 0, 1],
         [1, 1, 1],
         [0, 1, 1]]
    )
    color_all = [(1, 0, 0, 0.1),  # blue
                 (1, 1, 0, 0.1),  # yellow
                 (0, 1, 1, 0.1),  # cyan
                 (1, 0, 1, 0.1),  # magenta
                 (0.5, 0.5, 0.5, 0.1),  # gray
                 (0, 0.5, 0, 0.1)]  # green
    cube_size = np.asarray(cube_size)
    orig_point = np.asarray(orig_point)
    point_all = point_all_unit_cube * cube_size[np.newaxis, :]
    edge_all = [
        [point_all[0], point_all[1], point_all[2], point_all[3]],
        [point_all[0], point_all[1], point_all[5], point_all[4]],
        [point_all[1], point_all[2], point_all[6], point_all[5]],
        [point_all[2], point_all[3], point_all[7], point_all[6]],
        [point_all[3], point_all[0], point_all[4], point_all[7]],
        [point_all[4], point_all[5], point_all[6], point_all[7]]
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    face_all = Poly3DCollection(edge_all, linewidths=1, edgecolors='k')

    face_all.set_facecolor(color_all)

    ax.add_collection3d(face_all)

    # Plot the point_all themselves to force the scaling of the axes
    ax.scatter(point_all[:, 0], point_all[:, 1], point_all[:, 2], s=0)
    # ax.set_aspect('equal')
    axisEqual3D(ax)


if __name__ == '__main__':
    plot_cube([4, 6, 8])
    plt.show()