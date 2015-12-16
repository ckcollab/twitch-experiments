import av

from livestreamer import Livestreamer

from test_character_box import process_image

from PIL import Image


while True:
    session = Livestreamer()
    streams = session.streams('http://www.twitch.tv/anniebot')
    stream = streams['source']

    container = av.open(stream.url)
    video_stream = next(s for s in container.streams if s.type == b'video')

    image = None
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = frame.to_image()
            image.show()
            image.save("stream_great_1.png")
            #import ipdb; ipdb.set_trace()
            process_image(image)
            if image:
                # Just wait until we see dem in game 5 times
                break
        if image:
            break
    import time;time.sleep(1)
    # found = None
    # for packet in container.demux(video_stream):
    #     for frame in packet.decode():
    #         image = frame.to_image()
    #         image.show()
    #         import ipdb; ipdb.set_trace()
    #         found = process_image(image)
    #         if found:
    #             # Just wait until we see dem in game 5 times
    #             break
    #     if found:
    #         break

#
# fd = stream.open()
# data = ''
# while len(data) < 3e5:
#     data += fd.read(1024)
#     time.sleep(0.1)
# fd.close()

# fname = 'stream.bin'
# #open(fname, 'wb').write(data)
# capture = cv2.VideoCapture(fname)
# imgdata = capture.read()[1]
# #imgdata = imgdata[..., ::-1]  # BGR -> RGB
# img = Image.fromarray(imgdata)
# img.show()


# import av
#
# container = av.open('stream.bin')
# video = next(s for s in container.streams if s.type == b'video')
#
# for packet in container.demux(video):
#     for frame in packet.decode():
#         image = frame.to_image()
#         import ipdb; ipdb.set_trace()