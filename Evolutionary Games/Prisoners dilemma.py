import numpy as np
import matplotlib.pyplot as plt

N = 11      # Number of rounds
T = 0       # Punishment for lone snitch
R = 0.5     # Punishment if neither snitch
P = 1       # Punishment if both snitch
S = 1.5     # Punishment if snitched on

# Exercise 13.1
m = 6       # Opponents strategy
# n = 1       # Own strategy


def play_rounds(n, m, nrounds):
    m_punishment = 0
    o_punishment = 0
    m_prev = 0
    o_prev = 0
    for r in range(nrounds):
        m_play = make_choice(n, r, o_prev)
        o_play = make_choice(m, r, m_prev)

        if m_play == 0 and o_play == 0:
            m_punishment += R
            o_punishment += R
        if m_play == 0 and o_play == 1:
            m_punishment += S
            o_punishment += T
        if m_play == 1 and o_play == 0:
            m_punishment += T
            o_punishment += S
        if m_play == 1 and o_play == 1:
            m_punishment += P
            o_punishment += P
        m_prev = m_play
        o_prev = o_play
    return m_punishment, o_punishment


def make_choice(strat, r, other):
    if np.any(r+1 <= strat) and other == 0:
        return 0
    else:
        return 1

# # Playing for 1a:
# me1 = np.zeros(N)
# op1 = np.zeros(N)
#
# for n in range(N):
#     me1[n], op1[n] = play_rounds(n, m, N-1)
#
# # Playing for 1b:
# M = 11
# me2 = np.zeros([N, M])
# op2 = np.zeros([N, M])
#
# for m in range(M):
#     for n in range(N):
#         me2[n, m], op2[n, m] = play_rounds(n, m, N - 1)


# # Plotting for 1a:
# fig1a = plt.figure("Best strategy vs m=6", figsize=(12, 6))
# ax1a = fig1a.add_subplot(121)
#
# ax1a.scatter(range(N), me1)
# ax1a.set_xlabel("n")
# ax1a.set_ylabel("Years in prison")
# ax1a.set_title("Strategy evaluation vs m = 6")
#
# # 1b:
# ax1b = fig1a.add_subplot(122)
# im = ax1b.imshow(me2, origin="lower")
# ax1b.set_xlabel("m")
# ax1b.set_ylabel("n")
# ax1b.set_title("Years in prison")
# ax1b.figure.colorbar(im)
#
# # plt.savefig(f"figures/13_1c R=0,9 S=1,1")
# plt.show()

# 13.2
L = 3
strategies = np.ones([L, L])*(N-1)
strategies[1, 1] = 0
print(f"strats: \n{strategies}")
punishments = np.zeros(shape=strategies.shape)

for row, line in enumerate(strategies):
    for col, strat in enumerate(line):
        punishments[row,col], _ = play_rounds(row, col, N)

print(punishments)