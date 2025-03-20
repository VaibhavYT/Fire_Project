import cv2
import numpy as np
import time
import threading



initial_frame = None





video = cv2.VideoCapture('gettyimages-148739157-640_adpp (1).mp4')
time.sleep(2)  


width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_file = '/home/vaibhav/Desktop/Project/Fire_Project/fire_detection_output.mp4'
out = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))  

print(f"Saving output video to: {output_file}")

while True:
    
    ret, frame = video.read()
    if not ret:
        break
        
    
    frame = cv2.resize(frame, (640, 480))
    
    
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    
    
    if initial_frame is None:
        initial_frame = blur
        continue
        
    
    rgb_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    
    
    
    mask1 = cv2.inRange(rgb_frame, (124, 200, 220), (185, 255, 255))
    
    
    mask2 = cv2.inRange(rgb_frame, (0, 125, 220), (100, 200, 255))
    
    
    mask3 = cv2.inRange(rgb_frame, (0, 175, 220), (125, 255, 255))
    
    
    mask4 = cv2.inRange(hsv_frame, (0, 50, 50), (25, 255, 255))
    
    
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), cv2.bitwise_or(mask3, mask4))
    
    
    frame_delta = cv2.absdiff(initial_frame, blur)
    thresh = cv2.threshold(cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)[1]
    
    
    final_mask = cv2.bitwise_and(combined_mask, thresh)
    
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    fire_detected = False
    
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            fire_detected = True
    
    
    initial_frame = cv2.addWeighted(initial_frame, 0.9, blur, 0.1, 0)
    
    
    if fire_detected:
        cv2.putText(frame, "FIRE DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2)
        
    
    
    out.write(frame)
    
    
    cv2.imshow("Fire Detection", frame)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
out.release()  
cv2.destroyAllWindows()

print(f"Video saved to: {output_file}")
