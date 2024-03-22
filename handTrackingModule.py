import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handno=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand= self.results.multi_hand_landmarks[handno]

            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                #print (id,cx,cy)
                lmList.append([id,cx,cy])
                #if id == 4:
                if draw:
                    cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)

        return lmList
        
    def detect_gesture(lmList):
    # Calculate distances between specific landmarks
    thumb_tip = lmList[4]
    index_finger_tip = lmList[8]
    middle_finger_tip = lmList[12]
    ring_finger_tip = lmList[16]
    pinky_tip = lmList[20]

    # Calculate the Euclidean distances between finger tips
    dist_thumb_index = ((thumb_tip[1] - index_finger_tip[1]) ** 2 + (thumb_tip[2] - index_finger_tip[2]) ** 2) ** 0.5
    dist_thumb_pinky = ((thumb_tip[1] - pinky_tip[1]) ** 2 + (thumb_tip[2] - pinky_tip[2]) ** 2) ** 0.5

    # Detect gestures based on the distances
    if dist_thumb_index < 50 and dist_thumb_pinky < 50:
        return "Closed Fist"
    elif dist_thumb_index > 100 and dist_thumb_pinky > 100:
        return "Open Hand"
    else:
        return "Unknown Gesture"


def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        lmList = detector.findPosition(img)
        if (len(lmList)!=0):
            print(lmList[4])

        cTime = time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()

