import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
import sys
import logging

#LINE_START = sv.Point(320, 0)
#LINE_END = sv.Point(320, 480)


def main():
    #line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    
    )
    
    model = YOLO("yolov8m.pt")
    

    for result in model.track(source=0, show=False, stream=True, agnostic_nms=True,verbose=False):
        #print(result)
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        #detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]
        detections = detections[(detections.class_id == 39)]
        
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        #line_counter.trigger(detections=detections)
        #line_annotator.annotate(frame=frame, line_counter=line_counter)
       
        cv2.imshow("yolov8", frame)
        try:
            for detection_idx in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
                print("//////////////////////////////////////")
                print(f"classid: {detections.class_id[detection_idx]}  |  trackid: {detections.tracker_id[detection_idx]}")
                print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
                print(f"Center of the {model.model.names[detections.class_id[detection_idx]]} ---> x:{((x1+x2)/2).astype(int)} y:{((y1+y2)/2).astype(int)}")
                print("//////////////////////////////////////\n\n\n\n\n")
                
        except Exception as err:
            print("there is no object that YOLO can detect")
            print(f"Unexpected {err=}, {type(err)=}")

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()
