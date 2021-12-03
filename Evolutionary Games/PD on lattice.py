import numpy as np
import matplotlib.pyplot as plt





def play_rounds(n, m, nrounds):
    m_punishment = 0
    o_punishment = 0

    for r in range(nrounds):
        print(f"round {r}")
        m_play = make_choice(n, r)
        o_play = make_choice(m, r)

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

    return m_punishment # , o_punishment


def make_choice(strat, r):
    if r+1 <= strat:
        return 0
    else:
        return 1

N = 1      # Number of rounds
T = 0       # Punishment for lone snitch
R = 0.5     # Punishment if neither snitch
P = 1       # Punishment if both snitch
S = 1.5     # Punishment if snitched on

# 13.2
L = 5
strats = np.ones([L, L]) * (N - 0)
strats[2, 2] = 0
print(f"strats: \n{strats}")
punishments = np.zeros(shape=strats.shape)

for r, line in enumerate(strats):
    for c, strat in enumerate(line):
        print(strats)
        print(np.roll(strats, -1, axis=1))

        punishments[r, c] += play_rounds(strats[r, c], np.roll(strats, 1, axis=0)[r, c], N)
        punishments[r, c] += play_rounds(strats[r, c], np.roll(strats, -1, axis=0)[r, c], N)
        punishments[r, c] += play_rounds(strats[r, c], np.roll(strats, 1, axis=1)[r, c], N)
        punishments[r, c] += play_rounds(strats[r, c], np.roll(strats, -1, axis=1)[r, c], N)


print(f"Final punishments: \n{punishments}")



# def play_rounds(strats, nrounds):
#     punishmentgrid = np.zeros(shape=strats.shape)
#
#     for r in range(nrounds):
#         make_choice(strats, r)
#         # punishmentgrid +=
#
#
#     # print(punishmentgrid)
#
#
# def make_choice(strat, r):
#     print(r)
#     print(r < strat)    # coop
#
#
# # 13.2
# # at each time step each player plays the prisoners dilemma with 4 closest neighbors
# # if any neighbor achieves a better score, it updates strategy
#
# # Initialize lattice with random strategies for eahc site ranging so that
# # n in [o,N]
#
# # Competition: play with 4 von N neighbors
#
# # Revision: After playing, each player updates strategy to that with the best score
# # amongst those of its neighbor and itself. If its a tie, choose one randomly
#
# # Mutation: At the end of each time step, small prob mu for each player to
# # change to a random strategy
#
# # Single defector in the middle
# L = 3
# strategies = np.ones([L, L])*(N-1)
# strategies[1, 1] = 0
# print(f"strats: \n{strategies}")
#
#
#
#
# play_rounds(strategies, N)
