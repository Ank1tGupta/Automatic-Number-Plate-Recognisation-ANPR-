from ultralytics import YOLO
import cv2
import numpy as np

def resize_image_with_width(image, width):
    aspect_ratio = float(width) / image.shape[1]
    height = int(image.shape[0] * aspect_ratio)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return resized_image

# load models
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./test3.mp4')
# cap = cv2.VideoCapture('./sample.mp4')

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        frame_clone = frame.copy()  # Make a copy to draw on
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame_clone, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # process license plate

            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV)
            plate_processed = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            cv2.imshow("thresh", np.hstack((license_plate_crop_thresh,plate_processed)))

        cv2.imshow("Frame",resize_image_with_width(frame_clone,800))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
