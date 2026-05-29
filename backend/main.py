from ultralytics import YOLO
from api import traffic_data
import cv2
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open traffic video
cap = cv2.VideoCapture("videos/traffic.mp4")

# Vehicle class IDs
vehicle_classes = [2, 3, 5, 7]

# Create resizable window
cv2.namedWindow("Smart Traffic System", cv2.WINDOW_NORMAL)

# Signal sequence
signals = ["A", "B", "C", "D"]

# Current signal
current_signal_index = 0
current_signal = signals[current_signal_index]

# Signal state
signal_state = "GREEN"

# Timing
green_time = 15
yellow_time = 3

# Start timer
signal_start_time = time.time()

# Resize function
def resize_frame(frame, width=None):

    h, w = frame.shape[:2]

    if width is not None:
        ratio = width / w
        new_height = int(h * ratio)

        return cv2.resize(frame, (width, new_height))

    return frame


# Vehicle counting inside region
def count_vehicles(results, x1, y1, x2, y2):

    count = 0

    for box in results[0].boxes:

        # Skip if no tracking ID
        if box.id is None:
            continue

        # Get tracking ID
        track_id = int(box.id[0])

        # Get class ID
        cls = int(box.cls[0])

        # Ignore non-vehicles
        if cls not in vehicle_classes:
            continue

        # Get box coordinates
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])

        # Find center point
        center_x = (bx1 + bx2) // 2
        center_y = (by1 + by2) // 2

        # Check if vehicle belongs to region
        if x1 < center_x < x2 and y1 < center_y < y2:
            count += 1

    return count


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = resize_frame(frame, width=1200)

    # Get dimensions
    h, w = frame.shape[:2]

    # Run tracking
    results = model.track(frame, persist=True)

    # Draw detections
    annotated_frame = results[0].plot()

    # Define road regions
    road_A = (0, 0, w // 2, h // 2)
    road_B = (w // 2, 0, w, h // 2)
    road_C = (0, h // 2, w // 2, h)
    road_D = (w // 2, h // 2, w, h)

    # Count vehicles in each road
    count_A = count_vehicles(results, *road_A)
    count_B = count_vehicles(results, *road_B)
    count_C = count_vehicles(results, *road_C)
    count_D = count_vehicles(results, *road_D)

    road_counts = {
        "A": count_A,
        "B": count_B,
        "C": count_C,
        "D": count_D
    }

    # Time elapsed
    elapsed = int(time.time() - signal_start_time)

    # GREEN STATE
    if signal_state == "GREEN":

        remaining = green_time - elapsed

        if remaining <= 0:
            signal_state = "YELLOW"
            signal_start_time = time.time()

    # YELLOW STATE
    elif signal_state == "YELLOW":

        remaining = yellow_time - elapsed

        if remaining <= 0:

            # Move to next signal
            current_signal_index = (current_signal_index + 1) % 4
            current_signal = signals[current_signal_index]

            # AI decides timer ONCE
            current_count = road_counts[current_signal]

            if current_count <= 10:
                green_time = 15

            elif current_count <= 25:
                green_time = 30

            else:
                green_time = 50

            # Reset state
            signal_state = "GREEN"
            signal_start_time = time.time()


    # Update live dashboard data
    traffic_data["current_signal"] = current_signal
    traffic_data["signal_state"] = signal_state
    traffic_data["timer"] = remaining
    
    traffic_data["roads"]["A"] = count_A
    traffic_data["roads"]["B"] = count_B
    traffic_data["roads"]["C"] = count_C
    traffic_data["roads"]["D"] = count_D

    # Colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    GRAY = (80, 80, 80)
    WHITE = (255, 255, 255)

    # Draw road divisions
    cv2.rectangle(annotated_frame, (0, 0), (w // 2, h // 2), (255, 0, 0), 2)
    cv2.rectangle(annotated_frame, (w // 2, 0), (w, h // 2), (0, 255, 0), 2)
    cv2.rectangle(annotated_frame, (0, h // 2), (w // 2, h), (0, 0, 255), 2)
    cv2.rectangle(annotated_frame, (w // 2, h // 2), (w, h), (255, 255, 0), 2)

    # Highlight active road
    if current_signal == "A":
        cv2.rectangle(annotated_frame, (0, 0), (w // 2, h // 2), GREEN, 5)

    elif current_signal == "B":
        cv2.rectangle(annotated_frame, (w // 2, 0), (w, h // 2), GREEN, 5)

    elif current_signal == "C":
        cv2.rectangle(annotated_frame, (0, h // 2), (w // 2, h), GREEN, 5)

    elif current_signal == "D":
        cv2.rectangle(annotated_frame, (w // 2, h // 2), (w, h), GREEN, 5)

    # Vehicle counts on roads
    cv2.putText(
        annotated_frame,
        f"A: {count_A}",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        WHITE,
        3
    )

    cv2.putText(
        annotated_frame,
        f"B: {count_B}",
        (w // 2 + 40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        WHITE,
        3
    )

    cv2.putText(
        annotated_frame,
        f"C: {count_C}",
        (40, h // 2 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        WHITE,
        3
    )

    cv2.putText(
        annotated_frame,
        f"D: {count_D}",
        (w // 2 + 40, h // 2 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        WHITE,
        3
    )

    # Right dashboard panel
    cv2.rectangle(annotated_frame, (900, 0), (1200, h), (40, 40, 40), -1)

    # Dashboard title
    cv2.putText(
        annotated_frame,
        "SMART TRAFFIC SYSTEM",
        (910, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    # Signal display
    roads = ["A", "B", "C", "D"]

    start_y = 100

    for i, road in enumerate(roads):

        y = start_y + (i * 120)

        # Default signal colors
        red_light = RED
        yellow_light = GRAY
        green_light = GRAY

        # Active signal logic
        if road == current_signal:

            if signal_state == "GREEN":
                red_light = GRAY
                green_light = GREEN

            elif signal_state == "YELLOW":
                red_light = GRAY
                yellow_light = YELLOW

        # Road label
        cv2.putText(
            annotated_frame,
            f"Road {road}",
            (920, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WHITE,
            2
        )

        # Red light
        cv2.circle(annotated_frame, (1080, y - 20), 15, red_light, -1)

        # Yellow light
        cv2.circle(annotated_frame, (1080, y + 20), 15, yellow_light, -1)

        # Green light
        cv2.circle(annotated_frame, (1080, y + 60), 15, green_light, -1)

        # Vehicle count
        count = road_counts[road]

        cv2.putText(
            annotated_frame,
            f"Vehicles: {count}",
            (920, y + 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2
        )

    # Timer display
    cv2.putText(
        annotated_frame,
        f"TIMER: {remaining}",
        (940, 580),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        GREEN,
        3
    )

    # Current state
    cv2.putText(
        annotated_frame,
        f"STATE: {signal_state}",
        (920, 530),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2
    )

    # Show output
    cv2.imshow("Smart Traffic System", annotated_frame)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
