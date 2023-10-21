# importing necessary libraries
import cv2
import matplotlib.pyplot as plt

# loading config and pre-trained model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# loading classLabels from coco.names file
classLabels = [0] # empty list
file_name = 'coco.names'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    classLabels.append(fpt.read()) # adding classLabels to the end of the list
# setting up the model input parameters
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5) # normalizing input images
model.setInputMean((127.5, 127.5, 127.5)) # setting the mean value for the model
model.setInputSwapRB(True) # swapping blue and red color channels (OpenCV uses BGR instead of RGB)
# loading an image to detect objects in
img = cv2.imread('taha.jpg')
print(img.shape)

# plotting the image using Matplotlib's imshow
plt.imshow(img)

# displaying the image using OpenCV's imshow
cv2.imshow("Image", img)

# detecting objects in the image and printing the detected classIndices
classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
print(classIndex)

# plotting the detected objects in the image using OpenCV's rectangle and putText functions
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
    color=(0, 255, 0), thickness=3)

# plotting the final image with detected objects using Matplotlib's imshow function
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# displaying the final image with detected objects using OpenCV's imshow function
# detecting objects in a video stream
cap = cv2.VideoCapture("V.mp4")

# checking if the video file is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # raising an error if the video stream is not opened correctly
        raise print("cannot open video")

# detecting objects in each frame of the video stream
while True:
    # reading each frame of the video stream
    ret, frame = cap.read()
    # detecting objects in the frame and printing the detected classIndices
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    print(ClassIndex)
    # plotting the detected objects in the image
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                fontScale=font_scale, color=(0, 255, 0), thickness=3)
        cv2.imshow('Object Detection ', frame)
    # breaking the while loop if 'q' is pressed
    if cv2.waitKey(2) & 0XFF == ord('q'):
        break

# releasing the captured video stream and closing all windows
cap.release()
cv2.destroyAllWindows()