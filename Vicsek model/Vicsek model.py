import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull


N = 1000
L = 100
eta = 0.1
v = 1
dt = 1
R = [1]#, 2, 5, 10, 25, 50]
S = 10000
h = [0]
animation=False


def make_test_cfg():
    positions = np.array([[-30, -30],
                          [30, -30],
                          [-40, 0],
                          [0, -40]])
    orientations = np.array([np.pi / 4, 3 * np.pi / 4, 0, np.pi / 2])
    np.savetxt("collisionp.csv", positions, delimiter=",")
    np.savetxt("collisiono.csv", orientations, delimiter=",")


def make_starting_config():
    positions = np.random.uniform(low=-L / 2, high=L / 2, size=[N, 2])
    orientations = np.random.uniform(low=0, high=2 * np.pi, size=[N])
    np.savetxt("startpos2.csv", positions, delimiter=",")
    np.savetxt("startori2.csv", orientations, delimiter=",")


def make_past(N, h=3):
    positions = np.random.uniform(low=-L / 2, high=L / 2, size=[N, 2])
    orientations = np.random.uniform(low=0, high=2 * np.pi, size=[N])
    orientations = np.tile(orientations, (h,1))
    np.savetxt("pastp.csv", positions, delimiter=",")
    np.savetxt("pasto.csv", orientations, delimiter=",")


def read_configs(pos, ori):
    pos = np.genfromtxt(pos, delimiter=",")
    ori = np.genfromtxt(ori, delimiter=",")
    print(np.arctan(np.mean(np.sin(ori)) / np.mean(np.cos(ori))))
    return pos, ori


def main(N, L, eta, v, dt, R, S, h, animation):

    positions, orientations = read_configs("startpos2.csv", "startori2.csv")
    positions = np.random.uniform(low=-L / 2, high=L / 2, size=[N, 2])
    orientations = np.random.uniform(0, 2 * np.pi, size=[N])
    for r in R:
        for H in h:
            run_simulation(positions, orientations, N, L, eta, v, dt, r, S, H, animation)


def run_simulation(positions, orientations, N, L, eta, v, dt, R, S, h, animation=False):
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

    fig1 = plt.figure(f"Vicsek model, R = {R}", figsize=(12, 6))

    gcc = []
    gac = []

    # fig2 = plt.figure(f"Configuration after iterations")
    # ax2 = fig2.add_subplot()
    if animation:
        fig2 = plt.figure("Movement")
        axmov = fig2.add_subplot()

    for i in range(S):
        i += 1
        if i % 10 == 0:
            print(i)
        orientations = update_orientations(orientations, positions, eta, dt, R, N, h)
        velocities = get_velocities(orientations, v)
        positions = move(positions, velocities, dt)
        # if i == 1:
        #     print("mean vel: ", np.mean(velocities[:, 0]), np.mean(velocities[:, 1]))

        if animation:
            axmov.clear()
            axmov.set_xlim([-L / 2, L / 2])
            axmov.set_ylim([-L / 2, L / 2])
            axmov.scatter(positions[:, 0], positions[:, 1])
            plt.pause(0.0001)
            plt.show(block=False)

        gcc.append(global_clustering_coefficient(positions, R))
        gac.append(global_alignment_coefficient(velocities, v))

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
    ax6.legend(loc='best')

    fig1.savefig(f"Exercise 8.7, eta={eta}, R={R}, S={S}, N={N}, h={h}.png")
    plt.show()


def calc_pos(positions: np.ndarray):
    if positions.size == 2:
        print("ja", positions[0])
        if positions[0] < -L / 2:
            positions[0] += L
        elif positions[0] > L / 2:
            positions[0] -= L
        if positions[1] < -L / 2:
            positions[1] += L
        elif positions[1] > L / 2:
            positions[1] -= L
    else:
        for idx, pos in enumerate(positions):
            # positions[idx, 0] = pos[0]
            # positions[idx, 1] = pos[1]

            if positions[idx, 0] < -L / 2:
                positions[idx, 0] += L
            elif positions[idx, 0] > L / 2:
                positions[idx, 0] -= L
            if positions[idx, 1] < -L / 2:
                positions[idx, 1] += L
            elif positions[idx, 1] > L / 2:
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


def find_neighbors(positions, R, N):
    """
    :param positions:
    :param R:
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

        dists = np.sqrt((allpositions[:, 0] - pos1[0]) ** 2 + (allpositions[:, 1] - pos1[1]) ** 2)
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
        ids = np.mod(ids, N)
        # ids = np.unique(ids, axis=0)
        indices.append(ids)
    indices = np.array(indices, dtype=object)
    # print(indices)
    # indices = np.int(indices)
    #  neighbors,
    return indices


def find_k_neighbors(positions, k, N):
    """
    :param positions:
    :param R:
    :return indexes: Indices of neighbors, sorted and unique
    """
    # neighbors = np.empty([positions.shape[0],2])
    indices = []
    allpositions = duplicate_grid(positions)
    for idx1, pos1 in enumerate(positions):
        dists = np.sqrt((allpositions[:, 0] - pos1[0]) ** 2 + (allpositions[:, 1] - pos1[1]) ** 2)

        for i in range(k):
            ids = np.mod(np.argmin(dists), N)

        # ids = np.unique(ids, axis=0)
        indices.append(ids)
    indices = np.array(indices, dtype=object)
    # print(indices)
    # indices = np.int(indices)
    #  neighbors,
    return indices


def global_alignment_coefficient(velocities, v):
    N = np.shape(velocities)[0]
    return np.linalg.norm(np.sum(velocities / v, axis=0)) / N


def global_clustering_coefficient(positions, R):
    vor = Voronoi(positions)
    areas = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        cnt = 0
        if -1 in indices:

            areas[i] = np.inf
        else:
            areas[i] = ConvexHull(vor.vertices[indices]).volume

    cnt = (areas < np.pi * R ** 2).sum()
    return cnt / vor.npoints


def update_orientations(orientations, positions, eta, dt, R, N, h):
    indices = find_neighbors(positions, R, N)

    new_orientations = []
    for id, angle in enumerate(orientations):
        # mean_angle = np.mean(orientations[indices[id]])
        #
        # print(indices[id], type(indices[id]), type(indices[id][0]), type(id), len(indices[id]))
        # if len(indices[id]) == 1:
        #     # indices[id] = int(indices[id])
        #
        #     for i in indices[id]:
        #         mean_angle = 0
        #         mean_angle += np.arctan2(np.mean(np.sin(orientations[i])),
        #                                  np.mean(np.cos(orientations[i])))

        # print(" - ", orientations[indices[id]])
        # else:
        # print(id)
        # print(orientations[indices[id]])
        # print(np.sin(orientations[indices[id]]))
        # print(np.mean(np.sin(orientations[indices[id]])))

        mean_angle = np.arctan2(np.mean(np.sin(orientations[indices[id]])),
                                np.mean(np.cos(orientations[indices[id]])))
        W = np.random.uniform(-1 / 2, 1 / 2)
        new_orientations.append(mean_angle + eta * dt * W)
        # print("mean ang: ", mean_angle)
    return np.array(new_orientations)


def update_orientations2(orientations, positions, eta, dt, R, N, h):
    # Updating
    indices = find_neighbors(positions, R, N)

    orientations = orientations[0:h+1]
    new_orientations = []

    for id, angles in enumerate(orientations[0]): #in enumerate(orientations[0]):
        mean_angles = []
        mean_angles.append(np.arctan2(np.mean(np.sin(orientations[:, indices[id]])),
                                      np.mean(np.cos(orientations[:, indices[id]]))))
        mean = np.mean(mean_angles)
        W = np.random.uniform(-1 / 2, 1 / 2)
        new_orientations.append(mean + eta*dt*W)
    new_orientations = np.vstack((np.array(new_orientations), orientations[:-1]))
    # mean_angles = []
    # for id, angles in enumerate(orientations):
    #
    #     mean_angles.append(np.arctan2(np.mean(np.sin(orientations[row, indices[id]])),
    #                                       np.mean(np.cos(orientations[row, indices[id]]))))
    #     mean = np.mean(mean_angles)
    #     W = np.random.uniform(-1 / 2, 1 / 2)
    #     new_orientations.append(mean + eta*dt*W)
    # new_orientations = np.vstack((np.array(new_orientations), orientations[:-1]))

    return new_orientations



def get_velocities(orientations, v):
    if np.size(np.shape(orientations)) > 1:
        velocities = v * np.array([np.cos(orientations[0]), np.sin(orientations[0])])
    else:
        velocities = v * np.array([np.cos(orientations), np.sin(orientations)])
    # print("Vel: ", np.stack([velocities[0], velocities[1]], axis=1))
    return np.stack([velocities[0], velocities[1]], axis=1)
    # velocities = []
    # for idx, p in enumerate(orientations):
    #     velocities.append(v * np.array([np.cos(orientations[idx])*np.array([1, 0]) + np.sin(orientations[idx])*np.array([0,1])]))
    # print(velocities)
    # return np.array(velocities)


def move(oldpositions: np.ndarray, newvelocities: np.ndarray, dt):
    return calc_pos(oldpositions + newvelocities * dt)


# make_starting_config()

# make_test_cfg()
# make_past(N, h)
main(N, L, eta, v, dt, R, S, h, animation)
