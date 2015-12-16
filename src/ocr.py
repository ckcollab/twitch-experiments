import cv2
import numpy as np
import pytesseract

from PIL import Image, ImageDraw
from PIL import ImageFilter
from StringIO import StringIO


def ocr_file(file_name):
    image = Image.open(file_name)
    #image.show()
    return ocr_image(image)


def ocr_image(image):
    name_box = _get_character_name_box(image)
    return _ocr_name_box(name_box)


def _get_character_name_box(image):
    character_name_coords = _find_our_champion(image)
    #print "Found character name area:", character_name_coords
    if character_name_coords:
        character_name_image = image.crop(character_name_coords)
        #character_name_image.save("character_name_image_3.png")
        #character_name_image.show()
        return character_name_image


def _find_our_champion(image, search_step=2, draw_on_image=False):
    # should find the yellow bar underneath our hero and then we can use the top left of that
    # to center our search area
    data = np.asarray(image)
    data = data[:, :, :3]  # remove any alpha channel in the data so we just have (r, g, b)
    width, height = data.shape[1], data.shape[0]
    character_name_coords = None

    # Cut off the bottom few hundred pixels, not needed
    height -= 200

    for y in xrange(0, height):
        hits_this_row = 0
        for x in xrange(0, width, search_step):
            r, g, b = data[y][x]
            if r > 200 and 160 < g < 230 and 30 < b < 70:
                # really yellow
                hits_this_row += 1
            if hits_this_row > 20:
                character_name_coords = (x - 120, y - 35, x + 100, y - 8)  # Top left and bottom right
                if draw_on_image:
                    red = (255, 0, 0)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(character_name_coords, fill=red)
                break
        if character_name_coords:
            break
    return character_name_coords


def _ocr_name_box(name_box_image):
    try:
        gray = cv2.cvtColor(np.array(name_box_image), 0)  # 0 means load in grayscale
    except TypeError:
        gray = cv2.cvtColor(name_box_image, 0)  # 0 means load in grayscale
    ret, gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    #image = Image.fromarray(gray)
    #image.show()
    return pytesseract.image_to_string(Image.fromarray(gray))


if __name__ == "__main__":
    #ocr_file("/Users/eric/Downloads/stream_good_1.png")
    #ocr_file("/Users/eric/Downloads/stream_good_2.png")
    #ocr_file("/Users/eric/Downloads/stream_good_3.png")
    #ocr_file("/Users/eric/Downloads/stream_bad_1.png")
    #ocr_file("/Users/eric/Downloads/stream_bad_2.jpg")
    print "OCR'd stream_great_1.png: ", ocr_file("experiments/stream_great_1.png")
