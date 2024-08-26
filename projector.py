# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:06:01 2024

@author: sp3660
"""

# Receive input from an arduino and induce optogenetic stimulation based on the signal
# Instructions: - run standard Firmata on Arduino IDE (Examples -> firmata -> standard firmata) before running this script
#               - fill in the right dimensions and relative position of the second screen (the projector) and use fitting sized images
import sys # to access the system
import cv2
from time import sleep
import datetime
import pyfirmata    # Allows control of arduino in python
import matplotlib.pyplot as plt


# ============== params ============
INPUT_PIN = 0

win1_width = 500 # width of the main window, as offset for image (positive if the projector is right of screen, negative if left)
height_offset = 0 # difference in the height of the two windows (in Windows' settings)

stim_img_file = r"C:\Users\sp3660\Desktop\sign_map_10_5.png"
blank_img_file = r"C:\Users\sp3660\Desktop\sign_map_10_10.png"

arduino_port = "COM3"

log_file = "optoProjector_log.txt"

# =========== code ============

blue  = cv2.imread(stim_img_file, cv2.IMREAD_UNCHANGED)
black = cv2.imread(blank_img_file, cv2.IMREAD_UNCHANGED)

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)           # Create a virtual window named "image"
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Set as fullscreen
# cv2.moveWindow("image", win1_width, height_offset)          # move the window to the projector / second screen

# Start connection to arduino to read serial data
board = pyfirmata.Arduino(arduino_port)
it = pyfirmata.util.Iterator(board)
it.start()

LED = board.digital[13]     # Arduino Mega builtin LED

potentiometer = board.analog[INPUT_PIN]
potentiometer.enable_reporting()

opto_on = False
key = 0


def log(f, line):
    """
    This function will print and save logs
    """
    time = str(datetime.datetime.now())
    line = time + " " + line
    print(line)
    print(line, file=f, flush=True)


with open(log_file, 'a') as f:
    log(f, "Script started")

    while True:
        
        input_val = potentiometer.read()
        
        if input_val is None:
            input_val = 0
        
        if input_val > 0.5:
            do_opto = True
        else:
            do_opto = False
            
        if opto_on is False and do_opto is True:
            cv2.imshow("image", blue)
            LED.write(1)
            opto_on = True
            log(f, "projector on")

        if opto_on is True and do_opto is False:
            cv2.imshow("image", black)
            LED.write(0)
            opto_on = False
            log(f, "projector off")
        
        key = cv2.waitKey(1)    # Wait 1ms for cv2 to load the image

        if key == 27:           #if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break

sys.exit()
