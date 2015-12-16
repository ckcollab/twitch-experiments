import av
import sys
import time
import warnings

from livestreamer import Livestreamer

from experiments.test_detecting_in_lol_or_not import get_classifier, process_image
from ocr import ocr_image


# Hide warnings from SKLearn from flooding screen
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Please specify a streamer to monitor (python stream_frame_identifier.py day9)"
        exit(-1)

    streamer = sys.argv[1]

    classifier = get_classifier()
    is_in_lol = False

    while True:
        session = Livestreamer()
        streams = session.streams('http://www.twitch.tv/%s' % streamer)
        if streams:
            stream = streams['source']

            container = av.open(stream.url)
            video_stream = next(s for s in container.streams if s.type == b'video')

            image = None
            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    image = frame.to_image()
                    features = process_image(image)

                    # save our old state before checking new state, only show message when state changes
                    old_is_in_lol = is_in_lol

                    is_in_lol = classifier.predict(features)
                    if not old_is_in_lol and is_in_lol:
                        timestr = time.strftime("%Y%m%d-%I:%M %p")
                        print "@@@@@@@@@@ Joined game: %s" % timestr
                    elif old_is_in_lol and not is_in_lol:
                        timestr = time.strftime("%Y%m%d-%I:%M %p")
                        print "@@@@@@@@@@ Left game: %s" % timestr

                    if is_in_lol:
                        print "OCR from image, trying to read character name:", ocr_image(image)

                    # As soon as we get a full image, we're done
                    if image:
                        break

            time.sleep(1)
        else:
            print "Player not streaming, sleeping for 15 minutes"
            time.sleep(15 * 60)

