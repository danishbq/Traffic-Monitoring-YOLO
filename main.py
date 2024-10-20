import numpy as np
from ultralytics import YOLO
import cv2
import math
#import cvzone
from sort import *

#for videos
cap = cv2.VideoCapture("./Videos/sample-traffic.mp4")

model = YOLO("./YOLO-Weights/yolo11n.pt")

mask = cv2.imread("./mask.png")

#tracking unique vehicle in consecutive frames
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#limits for a point to count vehicles
limits = [380, 297, 673, 297]

totalCount = []
totalCar = 0
totalBike = 0
totalBus = 0
totalTruck = 0

while True:
    success, img = cap.read()
    if not success:
        break       #break loop when video ends

    imgRegion = cv2.bitwise_and(img,mask)

    results = model(imgRegion, stream=True)

    #array of detections for tracking
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            # Get the class name based on the class index 'cls'
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  # Lookup the class name

            if class_name == "car" or class_name == "truck" or class_name == "bus" or class_name == "motorbike" and conf > 0.3:
                print(x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
                # Combine class name and confidence for display
                label = f'{class_name}: {conf:.2f}'
                # Put the label above the bounding box for classification
                cv2.putText(img, label, (max(0,x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, tId = result
        x1, y1, x2, y2, tId = int(x1), int(y1), int(x2), int(y2), int(tId)
        print(result)
        cv2.putText(img, f'{tId}', (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        #finding center point of bounding boxes to check if it touches line to count
        cx, cy = (x1 + x2)//2, (y1+y2)//2
        cv2.circle(img,(cx,cy), 5, (255,255,255),cv2.FILLED)

        #limits = [380, 297, 673, 297]

        if limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[1]+15:
            if totalCount.count(tId) == 0:
                totalCount.append(tId)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                if class_name == "car":
                    totalCar+=1
                elif class_name == "motorbike":
                    totalBike+=1
                elif class_name == "truck":
                    totalTruck+=1
                elif class_name == "bus":
                    totalBus+=1



    cv2.putText(img, f' Count:{len(totalCount)} Cars:{totalCar} Bikes:{totalBike} Trucks:{totalTruck} Buses:{totalBus}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", img)
#    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)

# Calculate total vehicles and percentages
totalVehicles = len(totalCount)

if totalVehicles > 0:
    carPercentage = (totalCar / totalVehicles) * 100
    bikePercentage = (totalBike / totalVehicles) * 100
    truckPercentage = (totalTruck / totalVehicles) * 100
    busPercentage = (totalBus / totalVehicles) * 100
else:
    carPercentage = bikePercentage = truckPercentage = busPercentage = 0

# Display final statistics
print(f"\nTotal Vehicles: {totalVehicles}")
print(f"Cars: {totalCar} ({carPercentage:.2f}%)")
print(f"Motorbikes: {totalBike} ({bikePercentage:.2f}%)")
print(f"Trucks: {totalTruck} ({truckPercentage:.2f}%)")
print(f"Buses: {totalBus} ({busPercentage:.2f}%)")

# Optional: Display statistics on an image at the end
imgFinal = np.zeros((500, 800, 3), np.uint8)  # Create a black image for stats
cv2.putText(imgFinal, f'Total Vehicles: {totalVehicles}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(imgFinal, f'Cars: {totalCar} ({carPercentage:.2f}%)', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(imgFinal, f'Motorbikes: {totalBike} ({bikePercentage:.2f}%)', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(imgFinal, f'Trucks: {totalTruck} ({truckPercentage:.2f}%)', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(imgFinal, f'Buses: {totalBus} ({busPercentage:.2f}%)', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Final Statistics", imgFinal)
cv2.waitKey(0)
