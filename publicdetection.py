import cv2
import os
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
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            x, y = int(x), int(y)
                            people_boxes.append({'bbox': [x, y], 'score': detected_object.score})
                            break  # Only consider the first landmark
        return people_boxes

    @staticmethod
    def detect_objects(net, image, dim=300):
        blob = cv2.dnn.blobFromImage(image, size=(dim, dim), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()
        return detections


def main():
    classFile  = "coco_class_labels.txt"
    with open(classFile) as fp:
        labels = fp.read().split("\n")

    modelFile  = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
    configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

    # Load your model
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    person_detector = publicDetector()

    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        objects = publicDetector.detect_objects(net, frame, dim=300)

        num_people_detected = 0  # Counter for number of people detected

        for detection in objects[0,0,:,:]:
            score = float(detection[2])
            class_id = int(detection[1])
            if score > 0.5 and (labels[class_id] == 'person' or labels[class_id] == 'backpack'):
                num_people_detected += 1
                left = int(detection[3] * frame.shape[1])
                top = int(detection[4] * frame.shape[0])
                right = int(detection[5] * frame.shape[1])
                bottom = int(detection[6] * frame.shape[0])

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), thickness=10)
                label = f"{labels[class_id]}: {score:.2f}"
                cv2.putText(frame, label, (left, top-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1)

        people_boxes = person_detector.detect_people(frame)
        num_people_detected += len(people_boxes)
        for box in people_boxes:
            bbox = box['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'Number of people detected: {num_people_detected}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                     (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



