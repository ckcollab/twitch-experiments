import cv2
import numpy as np
import pytesseract

from PIL import Image, ImageDraw
from PIL import ImageFilter
from StringIO import StringIO



image = cv2.imread("character_name_image_1.png")

ret, thresh = cv2.threshold(image, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
x, y, w, h = cv2.boundingRect(cnt)
image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("images", np.hstack([image]))
cv2.waitKey(0)
