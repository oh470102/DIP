import matplotlib.pyplot as plt

def resolve_matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def live_plot(g1):

    resolve_matplotlib_error()
    
    plt.plot(g1)
    plt.grid(True)
    plt.draw()
    plt.gca().grid(True)
    plt.xlabel('episodes')
    plt.ylabel('scores')
    plt.pause(0.001)

def final_plot(g1):
    import numpy as np
    resolve_matplotlib_error()
    plt.ioff()

    window_size = 50
    moving_average = np.convolve(g1, np.ones(window_size) / window_size, mode='valid')

    plt.plot(moving_average, label='mAverage',color='red', linewidth=2.5)
    plt.legend()
    plt.draw()
    plt.show()