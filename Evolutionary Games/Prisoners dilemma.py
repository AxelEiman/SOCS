import numpy as np
import matplotlib.pyplot as plt

N = 11      # Number of rounds
T = 0       # Punishment for lone snitch
R = 0.5     # Punishment if neither snitch
P = 1       # Punishment if both snitch
S = 1.5     # Punishment if snitched on

# Exercise 13.1
m = 6       # Opponents strategy
n = 5       # Own strategy


def play_rounds(n, m, nrounds):
    m_punishment = 0
    o_punishment = 0
    m_prev = 0
    o_prev = 0
    for round in range(nrounds):
        m_play = make_choice(n, round, o_prev)
        o_play = make_choice(m, round, m_prev)

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


def make_choice(strat, round, other):
    if round+1 <= strat and other==0:
        return 0
    else:
        return 1

me = np.zeros(N)
op = np.zeros(N)
for n in range(N):
    print(f"n = {n}:")
    me[n], op[n] = play_rounds(n, m, N-1)
    print(f"Me: {me}, Opponent: {op}")

fig = plt.figure("Best strategy vs m=6")
ax = fig.add_subplot()

ax.scatter(range(N), me)
ax.set_xlabel("n")
ax.set_ylabel("Years in prison")
ax.set_title("Strategy evaluation vs m = 6")
plt.savefig("figures/13_1a")
plt.show()