import socket
import time 
import os
from pynput import keyboard  # For listening to keypresses

directory = r"C:\Codes\DP\dataset\terrains"

#ESP32_IP = "192.168.4.1"
ESP32_IP = "0.0.0.0"
UDP_PORT = 12345
timeout = 15

measuring_going = False

def count_files_with_substring(directory, substring):
    return sum(1 for filename in os.listdir(directory) if substring in filename)

def make_filename(terrain,pwm):
    substr = f"{terrain}_{pwm}"
    count = count_files_with_substring(directory=directory,substring=substr)
    filename = f"{directory}\\{terrain}_{pwm}_{count}.csv"
    return filename

# Function to send STOP command
def send_stop():
    global measuring_going
    message = "STOP"
    sock.sendto(message.encode(), ("192.168.4.1", UDP_PORT))
    print("Sent STOP command!")
    measuring_going = False

def send_start():
    message = "START"
    sock.sendto(message.encode(), ("192.168.4.1", UDP_PORT))
    print("Sent START command!")
    #measuring_going = True


# Background thread to listen for keypress (using `pynput`)
def on_press(key):
    try:
        if key.char == "s":  # If "s" is pressed, send stop signal
            send_stop()
    except AttributeError:
        pass  # Ignore special keys

pwm_set = False
pwm_value = 0
while not pwm_set:
    pwm_value = input("Enter a PWM duty cycle: ")
    pwm_value = int(pwm_value)

    if pwm_value > 0 and pwm_value <= 100:
        pwm_set = True

terrain = input("Enter a terrain type: ").lower()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock.settimeout(2)
sock.bind((ESP32_IP,UDP_PORT))
message = f"PWM:{pwm_value}"
recv_msg = ""
while recv_msg != "Y" and recv_msg != "y":
    sock.sendto(message.encode(),("192.168.4.1",UDP_PORT))
    #data,_ = sock.recvfrom(256)
    recv_msg = input("Type Y/y if the pwm is set correctly: ")
print("PWM sent")
measuring_going = True

filename = make_filename(terrain,pwm_value)#f"{terrain}_{pwm_value}.csv"
f = open(filename,"w")
input("Press enter to start measuring")

listener = keyboard.Listener(on_press=on_press)
listener.start()  # Run key listener in background

start_time = time.time()
current_time = start_time
print("Measuring started")
send_start()
while current_time - start_time < timeout:
    if not measuring_going:
        print("Measuring stopped with interrupt")
        break

    data,_ = sock.recvfrom(1024)
    message = data.decode()
    f.write(f"{current_time - start_time},{message}\n")
    current_time = time.time()
if current_time - start_time >= timeout:
    print("Measuring ended with timeout")

    send_stop()

print("Measuring ended")
#listener.close()
f.close()