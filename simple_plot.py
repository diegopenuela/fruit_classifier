import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()
ax.plot([1,2,3])
plt.draw()