import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(3)  

for i in range(60):
    ret, background = cap.read()
    if not ret:
        continue
background = np.flip(background, axis=1)  


lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

print("Background captured ✅")
print("Now wear your cloak and see the magic 🧙")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    cloak_mask = mask1 + mask2
    
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    cloak_mask = cv2.dilate(cloak_mask, np.ones((3,3), np.uint8), iterations=1)

    # Inverse mask
    inv_mask = cv2.bitwise_not(cloak_mask)

    # Segment cloak from background
    cloak_area = cv2.bitwise_and(background, background, mask=cloak_mask)
    rest_area = cv2.bitwise_and(frame, frame, mask=inv_mask)

    # Final output
    final_output = cv2.addWeighted(cloak_area, 1, rest_area, 1, 0)

    cv2.imshow("Invisible Cloak Effect", final_output)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# python -m venv .venv
# .venv\Scripts\activate
# pip install opencv-python numpy          
# python invisible_cloak.py
