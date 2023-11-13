from classes.audiobuffer import AudioBuffer
import PySimpleGUI as sg
import collections
import cv2
import sounddevice as sd
import multiprocessing

from ultralytics import YOLO

def main():

    audio_devices = {f"{audio_device['index']}: {audio_device['name']}":audio_device['index'] for audio_device in sd.query_devices()}


    audio_device_index = 0
    model_path = "model/yolov8n.pt"
    device = "CPU"
    max_n_people=13
    threshold = .20
    video_path = 0 #"audios/video.mp4" #
    layout = [
        [
            sg.Frame(
                "Settings",
                [
                    [
                        sg.Combo(list(audio_devices.keys()), default_value=list(audio_devices.keys())[0], readonly=True, enable_events=True, key="AUDIO_DEVICE"),
                        sg.Combo(["yolov8n.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], default_value="yolov8n.pt", key="MODEL", readonly=True, enable_events=True),
                        sg.Combo(["CPU", "CUDA", "MPS"], default_value="CPU", key=device, readonly=True, enable_events=True),
                        sg.Combo(list(range(1, max_n_people)), default_value=max_n_people, key="MAX_N_PEOPLE", readonly=True, enable_events=True),
                        sg.Slider(
                                    orientation="horizontal",
                                    key="THRESHOLD",
                                    range=(1, 100),
                                    default_value=20,
                                    enable_events=True
                                ),
                        sg.Combo(
                                    key="CAMERA",
                                    values=[
                                        "0",
                                        "1"
                                    ],
                                    readonly=True,
                                    default_value="0",
                                    enable_events=True
                                )
                        
                    
                    ],
                    [
                        sg.Button("Start", key="START")
                    ]
                ]
            )
        ]
    ]



    window = sg.Window("APARALIUS", layout)


    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == "Exit":
            break
        if event == "AUDIO_DEVICE":
            audio_device_index = audio_devices[values["AUDIO_DEVICE"]]
            sd.default.device = audio_device_index
        if event == "MODEL":
            model_path = f"model/{values['MODEL']}"
        if event == "DEVICE":
            device = values["DEVICE"].to_int()
        if event == "MAX_N_PEOPLE":
            max_n_people = values["MAX_N_PEOPLE"]
        if event == "CAMERA":
            video_path = values["CAMERA"]
        if event == "THRESHOLD":
            threshold = values["THRESHOLD"]/100
        if event == "START":

            # Load the YOLOv8 model
            model = YOLO(model_path) # yolov8n.pt, yolov8n-pose, yolov8n-seg

            # Open the video file
            
            cap = cv2.VideoCapture(video_path) # or 0 for webcam

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
                            conf=threshold,
                            half=False,
                            show=False,
                            save=False,
                            max_det=max_n_people,
                            classes=0,
                            verbose=False,
                            persist=True,
                            device=device,
                        )

                        # Get the boxes and track IDs
                        boxes = results[0].boxes.xywh.cpu()
                        audio_buffer.people_counter = min(len(boxes), max_n_people)
                        if results[0].boxes.id is not None:
                            track_ids = results[0].boxes.id.int().cpu().tolist()

                            # assign people counter and coords
                            list_a = {value: boxes[count] for count, value in enumerate(track_ids)}
                            
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
                        cv2.imshow("APARALIUS", annotated_frame)

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
                audio_buffer.join()
                break

            break
    window.close()
        
if __name__ == '__main__':

    # Pyinstaller fix
    multiprocessing.freeze_support()

    main()
