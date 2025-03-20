import cv2
import numpy as np
import time
import threading


initial_frame = None
fire_detected_counter = 0  
confidence_threshold = 3   
fire_confirmed = False     
confidence_threshold_percentage = 0.9  


from collections import deque
detection_history = deque(maxlen=10)  

video = cv2.VideoCapture('gettyimages-148739157-640_adpp (1).mp4')
time.sleep(2)  

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
    
    
    
    mask1 = cv2.inRange(rgb_frame, (150, 50, 50), (255, 120, 80))
    
    
    mask2 = cv2.inRange(rgb_frame, (180, 150, 30), (255, 255, 120))
    
    
    
    
    
    mask3 = cv2.inRange(rgb_frame, (200, 100, 0), (255, 230, 150))
    
    
    mask4 = cv2.inRange(hsv_frame, (0, 100, 150), (25, 255, 255))
    
    
    
    cloud_mask = cv2.inRange(hsv_frame, (0, 0, 200), (180, 30, 255))
    
    
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), cv2.bitwise_or(mask3, mask4))
    
    
    combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(cloud_mask))
    
    
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    
    
    frame_delta = cv2.absdiff(initial_frame, blur)
    thresh = cv2.threshold(cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)[1]
    
    
    final_mask = cv2.bitwise_and(combined_mask, thresh)
    
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    current_frame_fire_detected = False
    confidence_score = 0
    max_confidence = 0  
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 800:  
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h != 0 else 0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            
            
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            
            
            if roi.size == 0 or w < 3 or h < 3:
                continue
                
            
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv_roi[:,:,1])
            avg_value = np.mean(hsv_roi[:,:,2])
            
            
            b_var = np.var(roi[:,:,0])
            g_var = np.var(roi[:,:,1])
            r_var = np.var(roi[:,:,2])
            color_variance = (r_var + g_var + b_var) / 3
            
            
            cloud_penalty = 0
            if avg_saturation < 50:  
                cloud_penalty += 0.3
            if avg_value > 200:      
                cloud_penalty += 0.3
            if color_variance < 100:  
                cloud_penalty += 0.2
                
            
            shape_score = (1 - circularity) * 0.3  
            size_score = min(area / 5000, 1.0) * 0.4  
            ratio_score = (1 - abs(aspect_ratio - 1.5) / 1.5) * 0.15  
            solidity_score = (1 - solidity) * 0.15  
            
            
            confidence_score = shape_score + size_score + ratio_score + solidity_score - cloud_penalty
            
            
            if confidence_score > max_confidence:
                max_confidence = confidence_score
            
            
            if confidence_score > 0.5:
                
                if avg_saturation > 60 or color_variance > 200:  
                    current_frame_fire_detected = True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    confidence_percent = int(confidence_score * 100)
                    cv2.putText(frame, f"{confidence_percent}%", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    
    initial_frame = cv2.addWeighted(initial_frame, 0.95, blur, 0.05, 0)
    
    
    detection_history.append(current_frame_fire_detected)
    
    
    if current_frame_fire_detected:
        fire_detected_counter += 1
    else:
        fire_detected_counter = max(0, fire_detected_counter - 1)  
    
    if fire_detected_counter >= confidence_threshold:
        fire_confirmed = True
    elif fire_detected_counter == 0:
        fire_confirmed = False
    
    
    if fire_confirmed:
        if max_confidence > confidence_threshold_percentage:
            cv2.putText(frame, "FIRE DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
            cv2.putText(frame, f"Confidence: {max_confidence*100:.1f}%", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Potential Fire (Low Confidence)", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"Confidence: {max_confidence*100:.1f}%", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    
    history_count = sum(1 for x in detection_history if x)
    cv2.rectangle(frame, (10, 100), (10 + history_count * 20, 120), (0, 165, 255), -1)
    
    
    cv2.imshow("Fire Detection", frame)
    
    
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
