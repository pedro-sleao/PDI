import cv2
import os
from vision import detect_and_read_plate, count_vehicles, track_vehicles

def process_video(video_path, mode="vehicles", show=False):
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    last_plate = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Reduz resolução
        frame = cv2.resize(frame, (960, 540))

        # veiculos
        vehicle_count = None
        if mode in ("vehicles", "both"):
            frame, vehicle_count = count_vehicles(frame)

            cv2.putText(frame, f"Vehicles: {vehicle_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

        # placas
        if mode in ("plates", "both"):
            plate_text = ""

            frame, plate_text = detect_and_read_plate(frame)
            if plate_text:
                last_plate = plate_text

            if last_plate:
                cv2.putText(frame, f"Plate: {last_plate}",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)

        if show:
            cv2.imshow("Video Analysis", frame)
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_folder = "videos"

    for file in os.listdir(video_folder):
        if file.endswith((".mp4", ".avi", ".mov")):
            process_video(os.path.join(video_folder, file), mode="vehicles",show=True)
