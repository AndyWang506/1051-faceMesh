import cv2
import mediapipe as mp
import time

class FashMeshRecognition():
    def __init__(self, staticMode = False, maxFace = 1, minDetectConf= 0.5, minTrackConfi = 0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectConf = minDetectConf
        self.minTrackConfi = minTrackConfi

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectConf, self.minTrackConfi)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

        #
        #     # decrease the frame rate
        #     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     results = faceMesh.process(imgRGB)
        #     # Display results
        #     if results.multi_face_landmarks:
        #         for faceLandmarks in results.multi_face_landmarks:
        #             mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
        #             # get x,y,z position
        #             for id, landmark in enumerate(faceLandmarks.landmark):
        #                 # get the values of pixels, also mean convert landmarks to pixels
        #                 imageHeight, imageWeight, imageChannel = img.shape
        #                 x,y = int(landmark.x * imageWeight), int(landmark.y * imageHeight)
        #                 print(id, x,y)


def main():
    cap = cv2.VideoCapture("video/ai_generated_video.mp4")
    pTime = 0
    while True:
        success, img = cap.read()
        # create frame rate, cTime = current, pTime = previous
        cTime = time.time()
        fps = 1/ (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,70),cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()