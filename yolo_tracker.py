from collections import defaultdict
from classes.audiobuffer import AudioBuffer
import PySimpleGUI as sg
import collections
import cv2

from ultralytics import YOLO

layout = [
    [
        sg.Frame(
            "Settings",
            [[
                sg.Button("Start", key="START")
            ]]
        )
    ]
]


window = sg.Window("Video Tracking to Audio", layout)



while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Exit":
        break
    if event == "START":

        # Load the YOLOv8 model
        model = YOLO('model/yolov8n.pt') # yolov8n.pt, yolov8n-pose, yolov8n-seg

        # Open the video file
        video_path = "audios/video.mp4"
        cap = cv2.VideoCapture(0)
        max_n_people = 1

        success, frame = cap.read()
        if success:
            screen_height = frame.shape[0]
            scren_width = frame.shape[1]
            audio_buffer = AudioBuffer(scren_width, screen_height, max_n_people)
            audio_buffer.start()

            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                if success:
                    # Run YOLOv8 tracking on the frame, persisting tracks between frames
                    results = model.track(
                        frame,
                        tracker="botsort.yaml", # bytetrack.yaml
                        conf=0.4,
                        half=False,
                        show=False,
                        save=False,
                        max_det=max_n_people,
                        classes=0,
                        verbose=False,
                        persist=True,
                        device="cpu", # cpu cuda
                    )

                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    audio_buffer.people_counter = min(len(boxes), max_n_people)
                    if results[0].boxes.id is not None:
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # assign people counter and coords
                        list_a = {value:boxes[count] for count, value in enumerate(track_ids)}
                        
                        sorted_list = collections.OrderedDict(sorted(list_a.items()))
                        people_counter = 0
                        for box in sorted_list.values():
                            if people_counter >= max_n_people:
                                break
                            audio_buffer.stream_array[people_counter].coordinates.add_value(box)
                            people_counter += 1

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        audio_buffer.join()
                        break
                else:
                    # Break the loop if the end of the video is reached
                    break

            # Release the video capture object and close the display window
            cap.release()
            cv2.destroyAllWindows()

        break
window.close()
    
