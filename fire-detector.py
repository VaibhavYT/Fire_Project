import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import time
import os


os.environ['LD_PRELOAD'] = '/lib/x86_64-linux-gnu/libpthread.so.0'


root = tk.Tk()
root.withdraw()  

Fire_Reported = 0
last_alert_time = 0
alert_cooldown = 5  

def show_alert():
    global last_alert_time
    current_time = time.time()
    
    
    if current_time - last_alert_time > alert_cooldown:
        messagebox.showwarning("Fire Alert!", "Warning! A Fire Accident has been detected!")
        last_alert_time = current_time
        root.update()  


def get_camera():
    
    camera = cv2.VideoCapture('gettyimages-148739157-640_adpp (1).mp4')
    if camera is None or not camera.isOpened():
        
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if camera is None or not camera.isOpened():
        
        camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if camera is None or not camera.isOpened():
        
        camera = cv2.VideoCapture(0, cv2.CAP_GSTREAMER)
    return camera


video = get_camera()

if video is None or not video.isOpened():
    print("Error: Could not open video capture device")
    messagebox.showerror("Error", "Could not access the camera!")
    exit()

while True:
    
    root.update_idletasks()
    root.update()
    
    grabbed, frame = video.read()
    if not grabbed:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (960, 540))
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower = np.array([18, 50, 50], dtype="uint8")
    upper = np.array([35, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    no_fire_pixels = cv2.countNonZero(mask)

    if no_fire_pixels > 150:
        Fire_Reported += 1
        cv2.putText(frame, "Fire Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        show_alert()

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
root.destroy()  
