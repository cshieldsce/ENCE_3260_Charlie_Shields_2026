import serial
import matplotlib.pyplot as plt
import time

ser = serial.Serial("/dev/ttyACM0", 115200) # Adjust COM port and baud rate 
time.sleep(2) # Let serial connection come up

plt.ion() # Enablei nteractive mode
data = []

fig, ax = plt.subplots()
line, = ax.plot([],[])

ax.set_ylim(-10, 1100)
ax.set_xlim(0, 200)

last_plot_time = time.time()

try:
    while True:
        
        # exit if window closed
        if not plt.fignum_exists(fig.number):
            break

        # real serial
        line_bytes = ser.readline()

        try: 
            value = int(line_bytes.decode('utf-8', errors='ignore').strip())
        except:
            continue

        data.append(value)

        if len(data) > 200:
            data = data[-200:]

        # Display the analog value
        #print(value)
        if time.time() - last_plot_time > 0.05: # Update plot every 50 ms
            line.set_xdata(range(len(data)))
            line.set_ydata(data)
            plt.pause(0.05)
            last_plot_time = time.time()

        time.sleep(0.001) # Sleep to avoid flooding the console

except KeyboardInterrupt:
    print("\nExiting..")

finally:
    ser.close()