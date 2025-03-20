from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)
video = cv2.VideoCapture('gettyimages-148739157-640_adpp (1).mp4')
time.sleep(2)
initial_frame = None

def detect_fire():
    global initial_frame
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
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_fire(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)