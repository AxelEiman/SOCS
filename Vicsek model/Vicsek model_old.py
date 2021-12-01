import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull


N = 100
L = 100
eta = 0.01
v = 1
dt = 1
R = [1, 2, 5, 10, 25, 50]
S = 1000


def main(N, L, eta, v, dt, R, S):
    positions = np.random.uniform(low=-L/2, high=L/2, size=[N, 2])
    orientations = np.random.uniform(0, 2*np.pi, size=[N])

    for r in R:
        run_simulation(positions, orientations, N, L, eta, v, dt, r, S)


def run_simulation(positions, orientations, N, L, eta, v, dt, R, S):

    velocities = get_velocities(orientations, v)

    # print(neighbors, "\n")
    # for item in indexes:
    #     print(item)
    # print(indexes)

    # print(velocities)
    # print(global_alignment_coefficient(velocities, v))

    # for idx, row in enumerate(neighbors):
    #     print(positions[idx])
    #     print(row)

    fig1 = plt.figure(f"Vicsek model, R = {R}", figsize=(12,6))

    gcc = []
    gac = []
    for i in range(S):
        i += 1
        if i % 10 == 0:
            print(i)
        orientations = update_orientations(orientations, positions, eta, dt, R)
        velocities = get_velocities(orientations, v)
        positions = move(positions, velocities, dt)

        # ax1.clear()
        # ax1.set_xlim([-L / 2, L / 2])
        # ax1.set_ylim([-L / 2, L / 2])
        # ax1.scatter(positions[:, 0], positions[:, 1])

        gcc.append(global_clustering_coefficient(positions, R))
        gac.append(global_alignment_coefficient(velocities, v))

        # plt.pause(0.001)
        # if i in [10, 100, 500, 1000]:
        #     print(f"plotting {i}")
        #     fig2 = plt.figure(f"Configuration after {i} iterations")
        #     ax2 = fig2.add_subplot(23)
        #     vor = Voronoi(positions)
        #     voronoi_plot_2d(vor, ax2, show_vertices=False)
        #     # ax.scatter(positions[:,0], positions[:,1])
        #
        #     fig2.savefig(f"Iteration {i}.png")
        #     plt.show(block=False)
        if i == 10:
            ax1 = fig1.add_subplot(231)
            ax1.set_title(f"{i} iterations")
            vor = Voronoi(positions)
            voronoi_plot_2d(vor, ax1, show_vertices=False)
        if i == 100:
            ax2 = fig1.add_subplot(232)
            ax2.set_title(f"{i} iterations")
            vor = Voronoi(positions)
            voronoi_plot_2d(vor, ax2, show_vertices=False)
        if i == 500:
            ax3 = fig1.add_subplot(233)
            ax3.set_title(f"{i} iterations")
            vor = Voronoi(positions)
            voronoi_plot_2d(vor, ax3, show_vertices=False)
        if i == 1000:
            ax4 = fig1.add_subplot(234)
            ax4.set_title(f"{i} iterations")
            vor = Voronoi(positions)
            voronoi_plot_2d(vor, ax4, show_vertices=False)
        if i == 10000:
            ax5 = fig1.add_subplot(235)
            ax5.set_title(f"{i} iterations")
            vor = Voronoi(positions)
            voronoi_plot_2d(vor, ax5, show_vertices=False)

    ax6 = fig1.add_subplot(236)
    ax6.set_title("Clustering and alignment")
    ax6.plot(range(i), gcc, 'orange', label="Clustering")
    ax6.plot(range(i), gac, 'blue', label="Alignment")
    ax6.legend(loc='lower right')

    fig1.savefig(f"Exercise 8.4, eta={eta}, R={R}.png")
    plt.show(block=False)


def calc_pos(positions: np.ndarray):

    if positions.size == 2:
        if positions[0] < -L/2:
            positions[0] += L
        elif positions[0] > L/2:
            positions[0] -= L
        if positions[1] < -L/2:
            positions[1] += L
        elif positions[1] > L/2:
            positions[1] -= L
    else:
        for idx, pos in enumerate(positions):
            # positions[idx, 0] = pos[0]
            # positions[idx, 1] = pos[1]

            if positions[idx, 0] < -L/2:
                positions[idx, 0] += L
            elif positions[idx, 0] > L/2:
                positions[idx, 0] -= L
            if positions[idx, 1] < -L/2:
                positions[idx, 1] += L
            elif positions[idx, 1] > L/2:
                positions[idx, 1] -= L

    return positions


def duplicate_grid(positions):
    allpositions = np.append(positions, positions + [0, L], axis=0)
    allpositions = np.append(allpositions, positions + [L, L], axis=0)
    allpositions = np.append(allpositions, positions + [L, 0], axis=0)
    allpositions = np.append(allpositions, positions + [0, -L], axis=0)
    allpositions = np.append(allpositions, positions + [-L, -L], axis=0)
    allpositions = np.append(allpositions, positions + [-L, 0], axis=0)
    allpositions = np.append(allpositions, positions + [-L, L], axis=0)
    allpositions = np.append(allpositions, positions + [L, -L], axis=0)

    return allpositions


def find_neighbors(positions, R):
    """
    :param positions:
    :param R:
    :return neighbors: The positions of the neighbors, unsorted and with potential duplicates
    :return indexes: Indices of neighbors, sorted and unique
    """
    # neighbors = np.empty([positions.shape[0],2])
    indices = []
    allpositions = duplicate_grid(positions)
    for idx1, pos1 in enumerate(positions):
        # rowlist = []
        # ids = []
        # for idx, pos2 in enumerate(allpositions):
        #     if np.linalg.norm(pos2 - pos1) < R:
        #         # rowlist.append(calc_pos(pos2))
        #         ids.append(idx%50)

        dists = np.sqrt((allpositions[:,0] - pos1[0])**2 + (allpositions[:,1] - pos1[1])**2)
        # print(dists)
        ids = np.where(dists < R)[0]
        # rowlist = []
        # ids = []
        # for idx, pos2 in enumerate(allpositions):
        #     if np.linalg.norm(pos2 - pos1) < R:
        #         rowlist = np.append(rowlist, calc_pos(pos2))
        #         ids = np.append(ids, idx%50)

        # rowlist = np.array(rowlist)
        # neighbors = np.append(neighbors, rowlist, axis=0)
        ids = np.mod(ids, 50)
        # ids = np.unique(ids, axis=0)
        indices.append(ids)
    indices = np.array(indices, dtype=object)
    # print(indices)
        #  neighbors,
    return indices


def global_alignment_coefficient(velocities, v):
    N = np.shape(velocities)[0]
    return np.linalg.norm( np.sum( velocities/v , axis=0) )/N


def global_clustering_coefficient(positions, R):
    vor = Voronoi(positions)
    areas = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices:
            areas[i] = np.inf
        else:
            areas[i] = ConvexHull(vor.vertices[indices]).volume

            cnt = (areas < np.pi*R**2).sum()
    return cnt/vor.npoints


def update_orientations(orientations, positions, eta, dt, R):
    indices = find_neighbors(positions, R)  # TODO hitta orientation för alla nära.

    new_orientations = []
    for id, angle in enumerate(orientations):
        mean_angle = np.mean(orientations[indices[id]])
        new_orientations.append(mean_angle + eta*dt*np.random.uniform(-1/2,1/2))

    return np.array(new_orientations)


def get_velocities(orientations, v):
    velocities = v * np.array([np.cos(orientations), np.sin(orientations)])
    return np.stack([velocities[0], velocities[1]], axis=1)


def move(oldpositions: np.ndarray, newvelocities: np.ndarray, dt):
    return calc_pos(oldpositions + newvelocities*dt)


positions = np.random.uniform(low=-L/2, high=L/2, size=[N, 2])
orientations = np.random.uniform(0, 2*np.pi, size=[N])

main(N, L, eta, v, dt, R, S)