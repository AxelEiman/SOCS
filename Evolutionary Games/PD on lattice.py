import numpy as np
import matplotlib.pyplot as plt


def play_rounds(n, m, nrounds):
    m_punishment = 0
    o_punishment = 0

    for r in range(nrounds):
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


def compete(strats):
    punishments = np.zeros(shape=strats.shape)
    for r, line in enumerate(strats):
        for c, strat in enumerate(line):
            punishments[r, c] += play_rounds(strats[r, c], np.roll(strats, 1, axis=0)[r, c], N)
            punishments[r, c] += play_rounds(strats[r, c], np.roll(strats, -1, axis=0)[r, c], N)
            punishments[r, c] += play_rounds(strats[r, c], np.roll(strats, 1, axis=1)[r, c], N)
            punishments[r, c] += play_rounds(strats[r, c], np.roll(strats, -1, axis=1)[r, c], N)
    return punishments


def update_strats(s, p):
    # Check 4 neighbors punishments and copy best strat
    pl = np.roll(p, 1, axis=1)
    pr = np.roll(p, -1, axis=1)
    pu = np.roll(p, 1, axis=0)
    pd = np.roll(p, -1, axis=0)

    sl = np.roll(s, 1, axis=1)
    sr = np.roll(s, -1, axis=1)
    su = np.roll(s, 1, axis=0)
    sd = np.roll(s, -1, axis=0)

    for r, line in enumerate(p):
        for c, punishment in enumerate(p):
            strats = [s[r, c], sl[r, c], sr[r, c], su[r, c], sd[r, c]]
            punishments = [p[r, c], pl[r, c], pr[r, c], pu[r, c], pd[r, c]]
            id = np.argmin(punishments)

            s[r, c] = strats[id]
    return s


N = 11      # Number of rounds
T = 0       # Punishment for lone snitch
R = 0.5     # Punishment if neither snitch
P = 1       # Punishment if both snitch
S = 1.5     # Punishment if snitched on

# 13.2
L = 5
strats = np.ones([L, L]) * (N - 0)
strats[int(np.floor(L/2)), int(np.floor(L/2))] = 0
print(f"strats: \n{strats}")

punishments = compete(strats)

print(f"Punishments: \n{punishments}")

new_strats = update_strats(strats, punishments)
print(f"New strats: \n{new_strats}")

# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_title("strategies")
im1 = ax1.imshow(strats)
ax1.figure.colorbar(im1)

ax2 = fig.add_subplot(222)
im2 = ax2.imshow(punishments)
ax2.set_title("Punishments")
ax2.figure.colorbar(im2)

ax3 = fig.add_subplot(223)
im3 = ax3.imshow(new_strats)
ax3.set_title("New strats")
ax3.figure.colorbar(im3)

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
plt.show()


