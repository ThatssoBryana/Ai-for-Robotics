import cv2
import mediapipe as mp

class publicDetector():
    def __init__(self):
        self.mpObjectron = mp.solutions.objectron
        self.objectron = self.mpObjectron.Objectron(
            static_image_mode=False,
            max_num_objects=5,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def detect_people(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.objectron.process(image_rgb)

        people_boxes = []
        if results.detected_objects:
            for detected_object in results.detected_objects:
                if detected_object.label == 'Person':
                    for landmark in detected_object.landmarks_2d:
                        people_boxes.append({
                            'bbox': [int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])],
                            'score': detected_object.score})
                    break  

        return people_boxes, len(people_boxes)


def main():

    detector = publicDetector()

    # Gebruik de video camera van laptop
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        people_boxes, num_people = detector.detect_people(frame)


        for box in people_boxes:
            bbox = box['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Geef het aantal mensen detected
        cv2.putText(frame, f'Number of people: {num_people}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow('Frame', frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

