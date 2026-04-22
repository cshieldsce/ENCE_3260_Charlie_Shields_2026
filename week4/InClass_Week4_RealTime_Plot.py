import numpy as np 
import matplotlib.pyplot as plt
import time

plt.ion() # interactive mode on

t_data = []
y_data = []

start = time.time() # start our timer

while True:
    t = time.time() - start

    y = np.sin(2 * np.pi * 1 * t) + 0.8*np.random.random()
    
    t_data.append(t)
    y_data.append(y)

    # Filter noice
    #--------------------------------------#
    window = 20
    if (len(y_data) > window): 
        y_smooth = np.convolve(y_data, np.ones(window)/window, mode='same')
    else:
        y_smooth = y_data
    #--------------------------------------#

    plt.clf() # clear the frame
    plt.plot(t_data, y_data, alpha=0.3)
    plt.plot(t_data, y_smooth)
    
    #plt.legend()
    plt.xlim(max(0, t-5), t+1)
    plt.ylim(-1.5, 1.5)
    plt.title("Real-Time Signal")

    plt.pause(0.01) # 10 ms
    #plt.show()