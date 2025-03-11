import cv2
import numpy as np
import time
import threading
# playsound import removed

# Initial frame for background subtraction
initial_frame = None
# alarm_triggered variable removed

# play_alarm function removed

# Start video capture
video = cv2.VideoCapture(0)
time.sleep(2)  # Allow camera to initialize

while True:
    # Read frame
    ret, frame = video.read()
    if not ret:
        break
        
    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Create a blur to reduce noise
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    
    # If initial frame is None, initialize it
    if initial_frame is None:
        initial_frame = blur
        continue
        
    # Convert to different color formats
    rgb_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Create masks for different fire color ranges
    # RGB-based fire detection
    # Rule 1: R > 220, G > 200, 124 < B < 185
    mask1 = cv2.inRange(rgb_frame, (124, 200, 220), (185, 255, 255))
    
    # Rule 2: R > 220, 125 < G < 100, B < 100
    mask2 = cv2.inRange(rgb_frame, (0, 125, 220), (100, 200, 255))
    
    # Rule 3: R > 220, 175 < G < 125, B < 125
    mask3 = cv2.inRange(rgb_frame, (0, 175, 220), (125, 255, 255))
    
    # HSV-based fire detection (fire typically has hue in red-yellow range)
    mask4 = cv2.inRange(hsv_frame, (0, 50, 50), (25, 255, 255))
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), cv2.bitwise_or(mask3, mask4))
    
    # Calculate difference between current and initial frame for motion detection
    frame_delta = cv2.absdiff(initial_frame, blur)
    thresh = cv2.threshold(cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)[1]
    
    # Combine fire color detection with motion
    final_mask = cv2.bitwise_and(combined_mask, thresh)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize fire detected flag
    fire_detected = False
    
    # Process contours
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum area threshold
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            fire_detected = True
    
    # Update initial frame for background subtraction (adaptive)
    initial_frame = cv2.addWeighted(initial_frame, 0.9, blur, 0.1, 0)
    
    # Show current fire detection state
    if fire_detected:
        cv2.putText(frame, "FIRE DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2)
        # Alarm triggering code removed
    
    # Display the resulting frame
    cv2.imshow("Fire Detection", frame)
    # Fire Mask display removed
    
    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
