import sys 
import serial
import time
import threading
from queue import Queue
from collections import deque
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# Shared data queue
q = Queue()
running = True

# Serial reading thread
def read_serial(ser, q):
    global running

    while running:
        line_bytes = ser.readline()
        
        try:
            value = int(line_bytes.decode('utf-8', errors='ignore').strip())
            q.put(value)
        except:
            continue
# Qt App
app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(show=True, title="Real-time Serial Plot")

# Time-domain plot
plot_time  = win.addPlot(title="Time-Domain");
curve_raw  = plot_time.plot(pen=pg.mkPen((200, 200, 200)))
#curve_filt = plot_time.plot(pen=pg.mkPen('y', width=2))
plot_time.setYRange(-10, 1100)

# Data buffer
data = []

# Setup serial
ser = serial.Serial("/dev/ttyACM0", 115200) # Adjust COM port and baud rate 
time.sleep(2) # Let serial connection come up

# Create & start the serial thread
thread = threading.Thread(target=read_serial, args=(ser, q))
thread.start()

# Update plot
fs = 200 # Sampling freq

def update():
    global data
    
    max_per_update = 50 
    count = 0

    while not q.empty() and count < max_per_update:
        value = q.get()
        data.append(value)
        count += 1

    # Keep last 500 samples
    if len(data) > 500:
        data = data[-500:]

    if len(data) > 0:
        curve_raw.setData(data)

# Timer
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20) # Update every 20 ms

# Cleanup on close
def cleanup():
    global running
    running = False
    thread.join()
    ser.close()

app.aboutToQuit.connect(cleanup)

# Run
sys.exit(app.exec_())