import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# ORB detector
orb = cv2.ORB_create(2000)

# BF matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(prev_gray, None)

# Trajectory image
traj = np.zeros((600, 600, 3), dtype=np.uint8)

x, y = 300, 300  # starting position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)

    if des1 is not None and des2 is not None:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Take top matches
        matches = matches[:100]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Estimate motion using affine transform
        if len(pts1) > 10:
            M, _ = cv2.estimateAffinePartial2D(pts1, pts2)

            if M is not None:
                dx = M[0, 2]
                dy = M[1, 2]

                x += int(dx)
                y += int(dy)

                cv2.circle(traj, (x, y), 2, (0, 255, 0), -1)

    # Draw matches
    img_matches = cv2.drawMatches(prev_frame, kp1, frame, kp2, matches[:20], None)

    cv2.imshow("Matches", img_matches)
    cv2.imshow("Trajectory", traj)

    prev_gray = gray
    prev_frame = frame
    kp1, des1 = kp2, des2

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
