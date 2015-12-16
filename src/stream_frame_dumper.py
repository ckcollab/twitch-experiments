import av
import time

from livestreamer import Livestreamer


while True:
    session = Livestreamer()
    streams = session.streams('http://www.twitch.tv/Livibee')
    stream = streams['source']

    container = av.open(stream.url)
    video_stream = next(s for s in container.streams if s.type == b'video')

    image = None
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = frame.to_image()
            timestr = time.strftime("%Y%m%d-%H%M%S")
            image.save("stream_grabs/%s.png" % timestr)
            if image:
                break
        if image:
            break

    time.sleep(1)
