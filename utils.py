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
    plt.xlabel='episodes'
    plt.ylabel='scores'
    plt.pause(0.001)
