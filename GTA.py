import numpy as np
from PIL import ImageGrab
import cv2
import time
import math
import pyautogui
from numpy import ones, vstack
from numpy.linalg import lstsq
from control import PressKey, W, A, S, D
from statistics import mean


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros_like(img)
    img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, 540), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8,  line_img, 1.0, 0.0)
    return img


def pipeline(image):
    """
    An image processing pipeline which will output
    an image with the lane lines annotated.
    """
    height = image.shape[0]
    width = image.shape[1]
    #region_of_interest_vertices = np.array([[10, 200], [10,540], [200,540],[250,335],[480,300],[540,540], [770,540],[770,200],[550,200],[300,200]] ,np.int32)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #cropped_image2 = region_of_interest(gray_image, np.array([region_of_interest_vertices], np.int32),)
    canny_image = cv2.Canny(gray_image, 50, 150,apertureSize=3)
    final_image = cv2.GaussianBlur(canny_image, (5, 5), 0)
    lines = cv2.HoughLinesP(
        final_image,
        rho=1,
        theta=1*np.pi / 180,
        threshold=30,
     #   lines=np.array([]),
        minLineLength=20,
        maxLineGap=10  )
    left_line_x =  []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if math.fabs(slope) < 0.5:
                    continue
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
        min_y = int(image.shape[0] * (1.5 / 5))
        max_y = int(image.shape[0])
    except Exception as a:
        print("lel")
    try:
        poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
        ))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
    except Exception as e:
            print("lel")




    try:
        poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        line_image = draw_lines(
            final_image,
            [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
            ]],
            thickness=5,    )
        return line_image
    except Exception as f:
        print("lel")

    return final_image


last_time = time.time()
while True:
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    new_screen = pipeline(screen)
    cv2.imshow('window', new_screen)
    #cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break