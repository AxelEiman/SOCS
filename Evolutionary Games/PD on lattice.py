import numpy as np
import matplotlib.pyplot as plt


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
    return m_punishment # , o_punishment


def make_choice(strat, r, o_prev):
    if r+1 <= strat and o_prev == 0:
        return 0
    else:
        return 1


def compete(strats, nRounds):
    for round in range(nRounds):
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

    s_new = np.zeros(shape=s.shape)
    for r, line in enumerate(p):
        for c, punishment in enumerate(p):
            strats = [s[r, c], sl[r, c], sr[r, c], su[r, c], sd[r, c]]
            punishments = [p[r, c], pl[r, c], pr[r, c], pu[r, c], pd[r, c]]
            id = np.argmin(punishments)

            s_new[r, c] = strats[id]
    return s_new


def run_games(s, nRounds, timesteps, mu):
    counts = []
    for timestep in range(timesteps):
        # print(s)
        p = compete(s, nRounds)
        s = update_strats(s, p)
        s = mutate(s, mu, nRounds)
        counts.append(count_strats(s))
    return s, counts


def mutate(s, prob, N):
    r = np.random.rand(s.shape[0], s.shape[1])
    mask = r < prob
    # print(s)
    # print(mask)
    s[mask] = np.random.randint(0, N+1, s.shape)[mask]
    # print(s)
    # print(np.where(r < prob, s, np.random.randint(0, 2)*N))
    return s

def count_strats(s):
    unique, counts = np.unique(s, return_counts=True)
    # TODO göra detta till en matris typ? så kan man plotta varje linje som en rad kanske?
    return dict(zip(unique, counts))

N = 7      # Number of rounds
T = 0       # Punishment for lone snitch
R = 0.6     # Punishment if neither snitch
P = 1       # Punishment if both snitch
S = 1.5     # Punishment if snitched on
mu = 0.01   # Probability of
timesteps = 20

# 13.2
L = 30
# strats = np.ones([L, L]) * (N - 0)
# strats[int(np.floor(L/2)), int(np.floor(L/2))] = 0
# b)
# strats[6,6], strats[12,12], strats[18, 18], strats[24,24] = 0,0,0,0
# c)
# strats[int(np.floor(L/2)), int(np.floor(L/2))] = 0
# d)?
# strats[int(np.floor(L/2))+1, int(np.floor(L/2))] = N
# strats[int(np.floor(L/2))-1, int(np.floor(L/2))] = N
# strats[int(np.floor(L/2)), int(np.floor(L/2))+1] = N
# strats[int(np.floor(L/2)), int(np.floor(L/2))-1] = N

# 13.3
# strats = np.random.randint(0,2, [L,L])*N
# Rlist = np.linspace(0.8, 0.88, 20)
# Slist = np.linspace(1.3, 1.7, 20)

# coopRatio = []
# for i, S in enumerate(Slist):
#     print(f'{i} of {len(Slist)}')
#     strats_after = run_games(strats, N, timesteps, mu)
#     coopRatio.append(np.sum(strats_after)/(N*L**2))

# 13.4
strats = np.random.randint(0, N+1, [L,L])
fin_strats, amounts_list = run_games(strats, N, timesteps, mu)



# Plotting
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.set_title("strategies")
# im1 = ax1.imshow(strats)
# ax1.figure.colorbar(im1)

# ax2 = fig.add_subplot(222)
# im2 = ax2.imshow(punishments)
# ax2.set_title("Punishments")
# ax2.figure.colorbar(im2)

# ax3 = fig.add_subplot(122)
# im3 = ax3.imshow(strats_after)
# ax3.set_title("New strats")
# ax3.figure.colorbar(im3)

# ax1.axis('off')
# # ax2.axis('off')
# ax3.axis('off')

# Plotting 13.3
# fig = plt.figure()
# ax4 = fig.add_subplot()
# ax4.plot(Slist, coopRatio)
# ax4.set_ylabel("Share of cooperators")
# ax4.set_xlabel("R")

fig = plt.figure()
ax5 = fig.add_subplot(121)
im5 = ax5.imshow(fin_strats)
ax5.set_title("End strategies")
ax5.figure.colorbar()

ax6 = fig.add_subplot(122)



plt.show()


