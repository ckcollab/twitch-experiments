import cv2
import numpy as np
import pytesseract

from PIL import Image, ImageDraw
from PIL import ImageFilter
from StringIO import StringIO


def process_file(file_name):

    # # Finding blobs n shit
    # #
    # image = cv2.imread(file_name)
    # height, width = image.shape[:2]
    # print height, width
    # # Cut off the bottom 1/4th of the screen, causes false positives when looking for name
    # image = image[:-(height * .25), :]
    # # create NumPy arrays from the boundaries
    # lower = np.array([25, 146, 190], dtype="uint8")
    # upper = np.array([80, 220, 250], dtype="uint8")
    #
    # # find the colors within the specified boundaries and apply
    # #  the mask
    # mask = cv2.inRange(image, lower, upper)
    # output = cv2.bitwise_and(image, image, mask=mask)
    #
    #
    # # kernel = np.ones((1, 4), np.uint8)
    # # output = cv2.erode(output, kernel)
    # # kernel = np.ones((1, 4), np.uint8)
    # # output = cv2.erode(output, kernel)
    # # kernel = np.ones((1, 4), np.uint8)
    # # output = cv2.erode(output, kernel)
    #
    # output = cv2.resize(output, (output.shape[1]/3, output.shape[0]/3))
    #
    # # show the images
    #
    # #import ipdb; ipdb.set_trace()
    #
    # cv2.imshow("images", np.hstack([output]))
    # cv2.waitKey(0)
    # exit(0)







    print "\n\n\nReading file:", file_name
    image = Image.open(file_name)
    image.show()
    return process_image(image)


def process_image(image):
    character_name_coords = find_our_champion(image)
    print "Found character name area:", character_name_coords
    if character_name_coords:
        character_name_image = image.crop(character_name_coords)
        #character_name_image.save("character_name_image_3.png")
        character_name_image.show()
        return character_name_image


def find_our_champion(image, search_step=2, draw_on_image=False):
    # should find the yellow bar underneath our hero and then we can use the top left of that
    # to center our search area
    data = np.asarray(image)
    data = data[:, :, :3]  # remove any alpha channel in the data so we just have (r, g, b)
    width, height = data.shape[1], data.shape[0]
    character_name_coords = None

    #import ipdb; ipdb.set_trace()

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






if __name__ == "__main__":
    #process_file("/Users/eric/Downloads/stream_good_1.png")
    #process_file("/Users/eric/Downloads/stream_good_2.png")
    #process_file("/Users/eric/Downloads/stream_good_3.png")
    #process_file("/Users/eric/Downloads/stream_bad_1.png")
    #process_file("/Users/eric/Downloads/stream_bad_2.jpg")
    process_file("stream_great_1.png")
