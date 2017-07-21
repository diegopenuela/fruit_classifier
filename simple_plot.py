import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()
ax.plot([1,2,3])
plt.draw()