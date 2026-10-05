from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Create Deep SORT tracker
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    20.0,
    (640, 480)
)

# FPS variables
prev_time = time.time()

while True:
    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam")
        break

    # YOLO object detection
    results = model(frame, verbose=False)

    detections = []

    # Get YOLO detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            # Convert coordinates to Deep SORT format
            width = x2 - x1
            height = y2 - y1

            detections.append(
                ([x1, y1, width, height], confidence, class_id)
            )

    # Update Deep SORT tracker
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # Count detected objects
    object_count = len(detections)

    # Draw tracked objects
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_id = track.get_det_class()

        x1, y1, x2, y2 = track.to_ltrb()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Get class name
        class_name = model.names[int(class_id)]

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Draw label and tracking ID
        label = f"ID:{track_id} {class_name}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # Calculate FPS
    current_time = time.time()
    time_difference = current_time - prev_time

    if time_difference > 0:
        fps = 1 / time_difference
    else:
        fps = 0

    prev_time = current_time

    # Display object count
    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    out.write(frame)

    # Show output
    cv2.imshow(
        "CodeAlpha Object Detection & Tracking - Deep SORT",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()