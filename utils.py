import matplotlib.pyplot as plt

def resolve_matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def live_plot(g1, g2):
    resolve_matplotlib_error()
    plt.clf()
    plt.figure(dpi=200)
    plt.grid()

    plt.subplot(2,1,1)
    plt.plot(g1)
    plt.xlabel('epochs')
    plt.ylabel('score')

    plt.subplot(2,1,2)
    plt.plot(g2)
    plt.xlabel('timesteps')
    plt.ylabel('torque')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
