
from packages.centroidtracker import CentroidTracker
from packages.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

from flask import Flask, render_template, Response

app = Flask(__name__)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(r"modelfiles/MobileNetSSD_deploy.prototxt", r"modelfiles/MobileNetSSD_deploy.caffemodel")

if not (r"videos/example_01.mp4", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture("videos/example_01.mp4")



@app.route('/')
def index():
    return render_template('base.html')

def gen():
    writer = None

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    skip_frames = 30

    fps = FPS().start()
    while True:
        sucess ,frame = vs.read()

    	
        if "videos/example_01.mp4" is not None and frame is None:
            print("noframe")
            break
        
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
           
        status = "Waiting"
        rects = []
       
        if totalFrames % skip_frames == 0:
            print("detection")
            
            status = "Detecting"
            trackers = []
            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                print("first forloop")
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > 0.4:
                    print("yes greater")
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])
                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        print("yes person")
                        continue
                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                   
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    
                    trackers.append(tracker)
            
        else:
            
            for tracker in trackers:
                
                status = "Tracking"
                
                tracker.update(rgb)
                pos = tracker.get_position()
                
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                
                rects.append((startX, startY, endX, endY))
                
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        
        objects = ct.update(rects)
        
        for (objectID, centroid) in objects.items():
            
            to = trackableObjects.get(objectID, None)
            
            if to is None:
                to = TrackableObject(objectID, centroid)
               
            else:
               
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                # check to see if the object has been counted or not
                if not to.counted:
                    
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True
                    
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True
            
            trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.imwrite("3.jpg",frame)
            info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ("Status", status),
            ]
       
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # check to see if we should write the frame to disk
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                   bytearray(encodedImage) + b'\r\n')
        
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    vs.release()
    cv2.destroyAllWindows()