import cv2
from ultralytics import YOLO
import yolo_water as wrpr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = YOLO("best.pt")
    img= cv2.imread("test.jpg")
    msk = wrpr.get_yolo_bin_mask(model, img, 0)
    cnts = wrpr.get_yolo_contours(msk,area_thres=0.005)
    blank_image = np.zeros((msk.shape[0],msk.shape[1],3), np.uint8)
    if len(cnts) > 0:
        for cnt in cnts:
            prev_point = cnt[0,0,:]
            for pt in cnt[1:,0,:]:
                cv2.line(blank_image,prev_point, pt,(0,0,255),2)
                prev_point = pt
            cv2.line(blank_image,pt, cnt[0,0,:],(0,0,255),2)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(blank_image,cv2.COLOR_BGR2RGB))
        plt.savefig("res.png")
    else:
        print("No contours extracted!")