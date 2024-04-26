from ultralytics import YOLO
import torch 
import cv2
import supervision as sv


model = YOLO("/home/elscmp/Projects/pinger-nail-inspection/checkpoints/modest_tune.pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


results = model(source="/home/elscmp/Projects/pinger-nail-inspection/testcase/")


bb_annotator = sv.BoundingBoxAnnotator(
    color = sv.Color.RED,
    thickness=2,
)

lab_annotator = sv.LabelAnnotator(
    color = sv.Color.RED
)


for result in results:

    detections = sv.Detections.from_ultralytics(result)

    annotated_frame = bb_annotator.annotate(
        scene = result.orig_img.copy(),
        detections = detections
    )


    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]

    annotated_frame = lab_annotator.annotate(
        scene=annotated_frame,
        detections = detections,
        labels=labels
    )

    cv2.imshow("result", annotated_frame)
    cv2.waitKey()
    

