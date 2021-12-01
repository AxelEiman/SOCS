import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
#
#
# root = tkinter.Tk()
# root.wm_title("Embedding in Tk")
#
# fig = Figure(figsize=(5, 4), dpi=100)
# t = np.arange(0, 3, .01)
# ax = fig.add_subplot()
# line, = ax.plot(t, 2 * np.sin(2 * np.pi * t))
# ax.set_xlabel("time [s]")
# ax.set_ylabel("f(t)")
#
# canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
# canvas.draw()
#
# # pack_toolbar=False will make it easier to use a layout manager later on.
# toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
# toolbar.update()
#
# canvas.mpl_connect(
#     "key_press_event", lambda event: print(f"you pressed {event.key}"))
# canvas.mpl_connect("key_press_event", key_press_handler)
#
# button_quit = tkinter.Button(master=root, text="Quit", command=root.quit)
#
#
# def update_frequency(new_val):
#     # retrieve frequency
#     f = float(new_val)
#
#     # update data
#     y = 2 * np.sin(2 * np.pi * f * t)
#     line.set_data(t, y)
#
#     # required to update canvas and attached toolbar!
#     canvas.draw()
#
#
# slider_update = tkinter.Scale(root, from_=1, to=5, orient=tkinter.HORIZONTAL,
#                               command=update_frequency, label="Frequency [Hz]")
#
# # Packing order is important. Widgets are processed sequentially and if there
# # is no space left, because the window is too small, they are not displayed.
# # The canvas is rather flexible in its size, so we pack it last which makes
# # sure the UI controls are displayed as long as possible.
# button_quit.pack(side=tkinter.BOTTOM)
# slider_update.pack(side=tkinter.BOTTOM)
# toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
# canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
#
# tkinter.mainloop()

reach = np.random.rand(34,34)
fig3 = plt.figure("R as function of Beta and Beta/gamma")
ax3 = fig3.add_subplot()
a = ax3.imshow(reach)
yaxis = np.linspace(0,34,8)
ax3.set_yticks(yaxis)
ax3.set_yticklabels([10,20,30,40,50,60,70,80])
ax3.set_ylabel("Beta/gamma")

ax3.set_xlabel("Beta")
ax3.set_xticks([0,34/5, 2*34/5, 3*34/5,4*34/5,34])
ax3.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
fig3.colorbar(a)
ax3.invert_yaxis()
plt.show()