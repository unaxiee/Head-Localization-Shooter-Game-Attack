import torch
import numpy as np
from mss import mss
import cv2
import win32api, win32con
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='pre-s-best.pt')

monitor = {'left': 50, 'top': 230, 'width': 860, 'height': 645}

sct = mss()

while True:

    start_time = time.time()

    img = sct.grab(monitor)

    img = np.array(img)

    result = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if len(result.pred[0]) > 0:

        s = 0

        for i in range(len(result.pred[0])):

            x1 = int(result.pred[0][i][0].item())
            y1 = int(result.pred[0][i][3].item())
            x2 = int(result.pred[0][i][2].item())
            y2 = int(result.pred[0][i][1].item())
            x3 = int((x1 + x2) / 2)
            y3 = int((y1 + y2) / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.circle(img, (x3, y3), 3, (0,0,255), -1)

            s_tmp = (x2 - x1) * (y1 - y2)
            if s_tmp > s:
                x = x3 + 50
                y = y3 + 230

        #win32api.SetCursorPos((x, y))

        print('mouse position: (',x,y,')')

    end_time = time.time()

    fps = str(round(1 / (end_time - start_time), 3))

    cv2.putText(img, 'fps: ' + fps, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('monitor', img)

    #print(1 / (end_time - start_time))

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()