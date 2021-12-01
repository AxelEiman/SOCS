import numpy as np
from tkinter import *
from tkinter import ttk  # Is this needed?
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import time

# create root
res = 500
root = Tk()
hmarg = 1.1
vmarg = 1.3
root.geometry(f"{int(2 * res * hmarg)}x{int(res * vmarg)}")

# Animation window
canv = Canvas(root, bd=4, bg='#FFFFFF', relief=GROOVE)
root.lift()  # alt: .attributes('-topmost', 0)
canv.place(x=res / 20, y=res / 20, height=res, width=res)
ccolor = ['#34abeb', '#eb8334', '#17990b', "#000000"]

# Plot
# canv2 = Canvas(root, bd=4, bg='#FFFFFF', relief=GROOVE)
# canv2.place(x=2 * res / 20 + res, y=res / 20, height=res, width=res)

# example
fig = Figure(figsize=(5, 5), dpi=100)

ax = fig.add_subplot()

ax.set_xlabel("time steps")
ax.set_ylabel("Number of agents")


def restart():
    global S, t, S_oldold
    # Sort by x,y closest to center:
    I = np.random.randint(low=0, high=n, size=init_infected)
    S = np.zeros(n)
    S[I] = 1
    t = 0
    ax.clear()
    S_oldold = np.copy(S)
    canv2.draw()
    # run_sim()




canv2 = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
# canv2.draw()
toolbar = NavigationToolbar2Tk(canv2, root, pack_toolbar=False)
toolbar.update()


rest = Button(root, text="restart", command=restart)
rest.place(relx=0.025, rely=.85, relheight=0.12, relwidth=0.15)

Beta = Scale(root, from_=0, to=1, orient=HORIZONTAL, label='Infection probability', resolution=0.01)
Beta.place(relx=0.2, rely=.85, relheight=0.12, relwidth=0.15)
Beta.set(0.8)

Gamma = Scale(root, from_=0, to=1, orient=HORIZONTAL, label='Recovery rate', resolution=0.01)
Gamma.place(relx=0.35, rely=.85, relheight=0.12, relwidth=0.15)
Gamma.set(0.02)

Diff = Scale(root, from_=0, to=1, orient=HORIZONTAL, label='Diffusion probability', resolution=0.01)
Diff.place(relx=0.5, rely=.85, relheight=0.12, relwidth=0.15)
Diff.set(0.8)

# Parameters
n = 1000
init_infected = 10
lsize = 100

# System
x = np.floor(np.random.rand(n) * lsize)
y = np.floor(np.random.rand(n) * lsize)
I = np.random.randint(low=0, high=n, size=init_infected)
S = np.zeros(n)
S[I] = 1

nx = x  # temporary x
ny = y  # temporary y

particles = []
R = .5
for j in range(n):
    particles.append(canv.create_oval(x[j] * res / lsize,
                                      y[j] * res / lsize,
                                      (x[j] + 2 * R) * res / lsize,
                                      (y[j] + 2 * R) * res / lsize,
                                      outline=ccolor[0], fill=ccolor[0]))

timerCheck = time.perf_counter()
toolbar.pack(anchor="se")
canv2.get_tk_widget().pack(anchor="ne")

restart()

# For phase:
# b = np.linspace(0.01,1,34)
# Bg = np.linspace(1,80, 34)
# Normal:
b = np.array([0.8])
g = np.array([0.02])

# TODO skit i plotta alla enskilda plots, summera deaths och plotta som reach?
mu = 0.0025
reps = 1
reach = np.empty(shape=[np.size(g), np.size(b)])
dead = np.empty(shape=[np.size(g), np.size(b)])    # Kanske bara b ist.
for idb, beta in enumerate(b):
    print("b:", type(idb))
    # g = beta/Bg
    for idg, gamma in enumerate(g):
        print(type(idg))
        tmpReach = 0
        tmpDead = 0
        for i in range(reps):
            restart()
            while any(S == 1):
                t += 1
                if t % 100 == 0:
                    print(f'Elapsed time: {time.perf_counter()-timerCheck}')
                    timerCheck = time.perf_counter()
                # Phase:
                B = beta
                G = gamma
                # Normal:
                # B = Beta.get()
                # G = Gamma.get()
                D = Diff.get()

                # 2) Move with prob D:
                r = np.random.rand(n)
                steps_x = (r < D / 2)               # if lower than half, step x wise
                steps_y = (r > D / 2) & (r < D)     # if greater than half, step y wise == True

                nx = (x + np.sign(np.random.randn(n)) * steps_x) % lsize
                ny = (y + np.sign(np.random.randn(n)) * steps_y) % lsize

                # S_old = np.copy(S)

                # 3) Check infected agents and spread with prob. B:
                for i in np.where((S == 1) & (np.random.rand(n) < B))[0]:
                    S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1

                # 4) Recover with probability gamma:
                S[(S == 1) & (np.random.rand(n) < G)] = 2

                # Die with probability mu:
                S[(S == 1) & (np.random.rand(n) < mu)] = 3

                for j in range(n):
                    canv.move(particles[j], (nx[j] - x[j]) * res / lsize, (ny[j] - y[j]) * res / lsize)
                    canv.itemconfig(particles[j], fill=ccolor[int(S[j])], outline=ccolor[int(S[j])])

                if t % 20 == 0:

                    susceptible = ax.plot([t - 20, t], [np.count_nonzero(S_oldold == 0), np.count_nonzero(S == 0)], color=ccolor[0])
                    infected = ax.plot([t - 20, t], [np.count_nonzero(S_oldold == 1), np.count_nonzero(S == 1)], color=ccolor[1])
                    recovered = ax.plot([t - 20, t], [np.count_nonzero(S_oldold == 2), np.count_nonzero(S == 2)], color=ccolor[2])
                    dead = ax.plot([t-20, t], [np.count_nonzero(S_oldold == 3), np.count_nonzero(S == 3)], color=ccolor[3])
                    canv2.draw()
                    S_oldold = np.copy(S)
                #
                root.update()
                root.title(f"Susceptible: {sum(S == 0)}, Infected: {sum(S == 1)}, Recovered: {sum(S == 2)}, Dead: {sum(S ==3)}")

                x = nx
                y = ny
            tmpReach += np.count_nonzero(S == 2)
            tmpDead += np.count_nonzero(S == 3)
        # reach[idg, idb] = tmpReach/reps
        dead = tmpDead/reps

        print(f'Result: \n g={gamma}, b={beta}, R={reach[idg,idb]}, D = {dead}')
Tk.mainloop(root)

# fig2 = plt.figure("R as function of infection rate")
# ax2 = fig2.add_subplot()

# ax2.scatter(b/0.01, reach[0,:], color='blue', label="gamma = 0.01")
# ax2.scatter(b/0.02, reach[1,:], color='green', label="gamma = 0.02")
# ax2.set_xlabel("Beta/gamma")
# ax2.set_ylabel('R')
# ax2.legend(loc="best")


# fig3 = plt.figure("R as function of Beta and Beta/gamma")
# ax3 = fig3.add_subplot()
# ax3.matshow(reach)
# yaxis = np.linspace(10,80,8)
# ax3.set_xticks(xaxis)
# fig3 = plt.figure("R as function of Beta and Beta/gamma")
# ax3 = fig3.add_subplot()
# a = ax3.imshow(reach)
# yaxis = np.linspace(0,34,8)
# ax3.set_yticks(yaxis)
# ax3.set_yticklabels([10,20,30,40,50,60,70,80])
# ax3.set_ylabel("Beta/gamma")
#
# ax3.set_xlabel("Beta")
# ax3.set_xticks([0,34/5, 2*34/5, 3*34/5,4*34/5,34])
# ax3.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
# fig3.colorbar(a)
# ax3.invert_yaxis()
# plt.show()
# TODO plot 11.2 b)

