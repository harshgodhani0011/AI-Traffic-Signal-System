from flask import Flask, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import threading
import time
import numpy as np

app = Flask(__name__)
CORS(app)

# --------------------------------
# LIVE TRAFFIC DATA
# --------------------------------
traffic_data = {
    "current_signal": "A",
    "signal_state": "GREEN",
    "timer": 15,
    "roads": {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0
    }
}

output_frame = None

# --------------------------------
# API ROUTE
# --------------------------------
@app.route("/traffic-data")
def get_traffic_data():
    return jsonify(traffic_data)

# --------------------------------
# VIDEO STREAM
# --------------------------------
def generate_frames():

    global output_frame

    while True:

        if output_frame is None:
            continue

        _, buffer = cv2.imencode(
            '.jpg',
            output_frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# --------------------------------
# VIDEO FEED ROUTE
# --------------------------------
@app.route('/video-feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# --------------------------------
# PROCESS SINGLE ROAD
# --------------------------------
def process_road(frame, model):

    vehicle_classes = [2, 3, 5, 7]

    # Better YOLO settings
    results = model.track(
        frame,
        persist=True,
        conf=0.25,
        iou=0.5,
        imgsz=960
    )

    clean_frame = frame.copy()

    count = 0

    for box in results[0].boxes:

        cls = int(box.cls[0])

        if cls in vehicle_classes:
            count += 1

    return clean_frame, count

# --------------------------------
# MAIN AI SYSTEM
# --------------------------------
def run_ai_system():

    global output_frame

    # Better YOLO model
    model = YOLO("yolov8n.pt")

    # 4 videos
    cap_A = cv2.VideoCapture(
        "videos/roadA.mp4"
    )

    cap_B = cv2.VideoCapture(
        "videos/roadB.mp4"
    )

    cap_C = cv2.VideoCapture(
        "videos/roadC.mp4"
    )

    cap_D = cv2.VideoCapture(
        "videos/roadD.mp4"
    )

    signals = ["A", "B", "C", "D"]

    current_signal_index = 0
    current_signal = signals[0]

    signal_state = "GREEN"

    green_time = 15
    yellow_time = 3

    remaining = green_time

    last_timer_update = time.time()

    # --------------------------------
    # TRAFFIC HISTORY
    # --------------------------------
    history_A = []
    history_B = []
    history_C = []
    history_D = []

    # --------------------------------
    # STABLE TRAFFIC MEMORY
    # --------------------------------
    stable_counts = {
        "A": 10,
        "B": 10,
        "C": 10,
        "D": 10
    }

    last_detected_time = {
        "A": time.time(),
        "B": time.time(),
        "C": time.time(),
        "D": time.time()
    }

    while True:

        # ----------------------------
        # READ VIDEOS
        # ----------------------------
        retA, frameA = cap_A.read()
        retB, frameB = cap_B.read()
        retC, frameC = cap_C.read()
        retD, frameD = cap_D.read()

        # Restart videos
        if not retA:
            cap_A.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if not retB:
            cap_B.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if not retC:
            cap_C.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if not retD:
            cap_D.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize
        frameA = cv2.resize(frameA, (600, 350))
        frameB = cv2.resize(frameB, (600, 350))
        frameC = cv2.resize(frameC, (600, 350))
        frameD = cv2.resize(frameD, (600, 350))

        # ----------------------------
        # PROCESS ROADS
        # ----------------------------
        roadA, count_A = process_road(
            frameA,
            model
        )

        roadB, count_B = process_road(
            frameB,
            model
        )

        roadC, count_C = process_road(
            frameC,
            model
        )

        roadD, count_D = process_road(
            frameD,
            model
        )

        # ----------------------------
        # HISTORY SMOOTHING
        # ----------------------------
        history_A.append(count_A)
        history_B.append(count_B)
        history_C.append(count_C)
        history_D.append(count_D)

        history_A = history_A[-10:]
        history_B = history_B[-10:]
        history_C = history_C[-10:]
        history_D = history_D[-10:]

        avg_A = int(sum(history_A) / len(history_A))
        avg_B = int(sum(history_B) / len(history_B))
        avg_C = int(sum(history_C) / len(history_C))
        avg_D = int(sum(history_D) / len(history_D))

        road_counts = {
            "A": avg_A,
            "B": avg_B,
            "C": avg_C,
            "D": avg_D
        }

        # ----------------------------
        # STABLE TRAFFIC SYSTEM
        # ----------------------------
        current_time = time.time()

        for road in road_counts:

            detected_count = road_counts[road]

            if detected_count > 0:

                stable_counts[road] = detected_count

                last_detected_time[road] = current_time

            else:

                # Keep previous value for 5 sec
                if (
                    current_time -
                    last_detected_time[road]
                ) < 5:

                    road_counts[road] = stable_counts[road]

                else:

                    # Minimum traffic
                    road_counts[road] = 1

        # ----------------------------
        # SMOOTH TIMER SYSTEM
        # ----------------------------
        if current_time - last_timer_update >= 1:

            remaining -= 1

            last_timer_update = current_time

        # Prevent negative timer
        if remaining < 0:
            remaining = 0

        # ----------------------------
        # GREEN STATE
        # ----------------------------
        if signal_state == "GREEN":

            if remaining <= 0:

                signal_state = "YELLOW"

                remaining = yellow_time

        # ----------------------------
        # YELLOW STATE
        # ----------------------------
        elif signal_state == "YELLOW":

            if remaining <= 0:

                current_signal_index = (
                    current_signal_index + 1
                ) % 4

                current_signal = signals[
                    current_signal_index
                ]

                current_count = road_counts[
                    current_signal
                ]

                # --------------------------------
                # GUARANTEED MINIMUM TIMER
                # --------------------------------
                if current_count <= 10:

                    green_time = 15

                elif current_count <= 20:

                    green_time = 30

                else:

                    green_time = 50

                signal_state = "GREEN"

                remaining = green_time

                last_timer_update = time.time()

        # ----------------------------
        # UPDATE DASHBOARD
        # ----------------------------
        traffic_data["current_signal"] = current_signal
        traffic_data["signal_state"] = signal_state
        traffic_data["timer"] = remaining

        traffic_data["roads"]["A"] = road_counts["A"]
        traffic_data["roads"]["B"] = road_counts["B"]
        traffic_data["roads"]["C"] = road_counts["C"]
        traffic_data["roads"]["D"] = road_counts["D"]

        # ----------------------------
        # TEXT OVERLAY
        # ----------------------------
        cv2.putText(
            roadA,
            f"Road A: {road_counts['A']}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            roadB,
            f"Road B: {road_counts['B']}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            roadC,
            f"Road C: {road_counts['C']}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            roadD,
            f"Road D: {road_counts['D']}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # ----------------------------
        # ACTIVE SIGNAL BORDER
        # ----------------------------
        GREEN = (0,255,0)

        if current_signal == "A":

            cv2.rectangle(
                roadA,
                (0,0),
                (599,349),
                GREEN,
                8
            )

        elif current_signal == "B":

            cv2.rectangle(
                roadB,
                (0,0),
                (599,349),
                GREEN,
                8
            )

        elif current_signal == "C":

            cv2.rectangle(
                roadC,
                (0,0),
                (599,349),
                GREEN,
                8
            )

        elif current_signal == "D":

            cv2.rectangle(
                roadD,
                (0,0),
                (599,349),
                GREEN,
                8
            )

        # ----------------------------
        # CREATE 2x2 GRID
        # ----------------------------
        top_row = np.hstack(
            (roadA, roadB)
        )

        bottom_row = np.hstack(
            (roadC, roadD)
        )

        final_frame = np.vstack(
            (top_row, bottom_row)
        )

        output_frame = final_frame.copy()

# --------------------------------
# START AI THREAD
# --------------------------------
threading.Thread(
    target=run_ai_system,
    daemon=True
).start()

# --------------------------------
# RUN FLASK
# --------------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )