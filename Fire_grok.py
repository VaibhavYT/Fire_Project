import cv2
import numpy as np

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_subtractor.apply(gray)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire1 = np.array([0, 50, 100])
    upper_fire1 = np.array([40, 255, 255])
    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)

    lower_fire2 = np.array([170, 50, 100])
    upper_fire2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)

    total_color_mask = mask1 + mask2
    fire_mask = cv2.bitwise_and(fg_mask, total_color_mask)

    kernel = np.ones((5,5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        print("Fire detected")
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        print("No fire")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()