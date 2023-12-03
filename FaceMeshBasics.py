import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video/ai_generated_video.mp4")

pTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)           # Go to -> implementation & look its input for def __init__
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)    # Drawing Specifications

while True:
    success, img = cap.read()
    # decrease the frame rate
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # Display results
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            # get x,y,z position
            for id, landmark in enumerate(faceLms.landmark):
                # get the values of pixels, also mean convert landmarks to pixels
                imageHeight, imageWeight, imageChannel = img.shape
                x,y = int(landmark.x * imageWeight), int(landmark.y * imageHeight)
                print(id, x,y)

    # create frame rate, cTime = current, pTime = previous
    cTime = time.time()
    fps = 1/ (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
