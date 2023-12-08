import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe as mp
import time
import math

class FaseMeshRecognition():
    def __init__(self, staticMode=False, maxFace=3, minDetectConf=0.5, minTrackConfi=0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectConf = minDetectConf
        self.minTrackConfi = minTrackConfi

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
#       self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectConf, self.minTrackConfi)
# This is for old Mediapipe version, using this will cause TypeError: create_bool(): incompatible function arguments.
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFace,
            min_detection_confidence=self.minDetectConf,
            min_tracking_confidence=self.minDetectConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    # Find the distance between two landmarks based on their index numbers.
    # Application -> when blinking our eyes, the values should be smaller than usual
    def findDistance(self,p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info


    def findFaceMesh(self, img, draw=True):

        # decrease the frame rate
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        # Display results
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face = []
                # get x,y,z position
                for id, landmark in enumerate(faceLms.landmark):
                    # get the values of pixels, also mean convert landmarks to pixels
                    imageHeight, imageWeight, imageChannel = img.shape
                    x, y = int(landmark.x * imageWeight), int(landmark.y * imageHeight)
                    # Display all 468 landmarks on our face
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
                    print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces




def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    recognition = FaseMeshRecognition()
    while True:
        success, img = cap.read()
        # create frame rate, cTime = current, pTime = previous
        # need to add 'False' if we want to display 468 landmarks (cv2.putText)
        # we can add 'False' to remove the white landmarks and lines
        img, faces = recognition.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces[0]))
        if faces:
            for index in faces:
                leftEyeUpPoint = index[159]
                leftEyeDownPoint = index[23]
                leftEyeVerticalDistance, info = recognition.findDistance(leftEyeUpPoint, leftEyeDownPoint)
                print(leftEyeVerticalDistance)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
