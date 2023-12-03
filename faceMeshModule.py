import cv2
import mediapipe as mp
import time


class FashMeshRecognition():
    def __init__(self, staticMode=False, maxFace=1, minDetectConf=0.5, minTrackConfi=0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectConf = minDetectConf
        self.minTrackConfi = minTrackConfi

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectConf, self.minTrackConfi)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

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
                    x,y = int(landmark.x * imageWeight), int(landmark.y * imageHeight)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture("video/ai_generated_video.mp4")
    pTime = 0
    recognition = FashMeshRecognition()
    while True:
        success, img = cap.read()
        # create frame rate, cTime = current, pTime = previous
        img, faces = recognition.findFaceMesh(img, False)
        if len(faces) != 0:
            print(len(faces[0]))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
