import cv2
import numpy as np
import time
import pandas as pd

cv2.getBuildInformation()
# Load the YOLOv4 model
model = cv2.dnn.readNetFromDarknet('/Users/MacbookPro/Downloads/yolov4.cfg', '/Users/MacbookPro/Downloads/yolov4.weights')
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the class labels
with open('/Users/MacbookPro/Downloads/coco.names', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Create a dataframe to store the object labels and timestamps
object_data = pd.DataFrame(columns=['Label', 'Timestamp'])


# Open the video file
cap = cv2.VideoCapture('/Users/MacbookPro/Downloads/Traffic Lights.mp4')

# Loop over the frames in the video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get the current timestamp
    timestamp = time.time()

    # Create a blob from the image and perform forward pass on the YOLOv4 model
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True)
    model.setInput(blob)
    outputs = model.forward(model.getUnconnectedOutLayersNames())

    # Loop over the detected objects and draw bounding boxes
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                left = int(center_x - width/2)
                top = int(center_y - height/2)

                # Add the object label and timestamp to the dataframe
                label = labels[class_id]
                print(label)
                
                object_data.loc[len(object_data)] = [label, timestamp]

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    #exit()
    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Save the object data to a CSV file
object_data.to_csv('/Users/MacbookPro/Downloads/object_data.csv', index=False)

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
