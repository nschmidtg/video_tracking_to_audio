from collections import defaultdict
from classes.audiobuffer import AudioBuffer
from classes.settings import Settings
import operator
import PySimpleGUI as sg
import sys

import cv2
import numpy as np

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



window = sg.Window("Image to MIDI", layout)


# audio_buffer.start()
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Exit":
        break
    if event == "START":

        # Load the YOLOv8 model
        model = YOLO('model/yolov8n.pt')

        # Open the video file
        video_path = "path/to/video.mp4"
        cap = cv2.VideoCapture(0)

        # Store the track history
        track_history = defaultdict(lambda: [])
        max_n_people = 2

        success, frame = cap.read()
        if success:
            print("frame shapee", frame.shape)
            screen_height = frame.shape[0]
            scren_width = frame.shape[1]
            settings = Settings(
                scren_width,
                screen_height,
                max_n_people
            )
            
            audio_buffer = AudioBuffer(settings)
            audio_buffer.start()

            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                if success:
                    # Run YOLOv8 tracking on the frame, persisting tracks between frames
                    results = model.track(
                        frame,
                        persist=True,
                        tracker="botsort.yaml",
                        conf=0.2,
                        imgsz=640,
                        half=False,
                        show=True,
                        save=False,
                        max_det=5,
                        vid_stride=True,
                        classes=0,
                    )

                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    if results[0].boxes.id is not None:
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # Visualize the results on the frame
                        annotated_frame = results[0].plot()

                        # assign people counter and coords
                        settings.people_counter = min(len(boxes), max_n_people)
                        
                        sorted_list = sorted(list(zip(track_ids, boxes)), key=operator.itemgetter(1))
                        people_counter = 0
                        for _, box in sorted_list:
                            if people_counter >= max_n_people:
                                break
                            settings.coords[people_counter] = box
                            people_counter += 1

                        # Display the annotated frame
                        cv2.imshow("YOLOv8 Tracking", annotated_frame)

                        # Break the loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            settings.keep_playing = False
                            break
                else:
                    # Break the loop if the end of the video is reached
                    break

            # Release the video capture object and close the display window
            cap.release()
            cv2.destroyAllWindows()
        break
window.close()
    
