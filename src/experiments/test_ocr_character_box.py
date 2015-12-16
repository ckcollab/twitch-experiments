import pytesseract

from PIL import Image, ImageFilter

import cv2
import numpy as np

gray = cv2.imread("character_name_image_1.png", 0)  # 0 means load in grayscale
#gray = cv2.imread("character_name_image_2.png", 0)  # 0 means load in grayscale
ret, gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
image = Image.fromarray(gray)
#image.save("character_name_image_processed_2.png")
image.show()
print "OCR'd:", pytesseract.image_to_string(image)
# cv2.imshow("images", np.hstack([gray]))
# cv2.waitKey(0)
#import ipdb; ipdb.set_trace()







# image = Image.open("character_name_image_1.png")
# # image = image.filter(ImageFilter.SHARPEN)
# # image = image.filter(ImageFilter.SHARPEN)
# # image = image.filter(ImageFilter.SHARPEN)
# # image = image.filter(ImageFilter.SHARPEN)
# image.show()
# print "OCR'd:", pytesseract.image_to_string(image)






# Turn the image negative or something, make the white text really stand out
# I think we can get it this way weeeeeeeeeeee








