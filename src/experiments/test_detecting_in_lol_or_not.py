'''
Based off of this day/night detection script:
http://www.ippatsuman.com/2014/08/13/day-and-night-an-image-classifier-with-scikit-learn/
'''
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import urllib2
import sys
import os

from glob import glob
from PIL import Image, ImageFilter
from progressbar import ProgressBar, Bar, ETA
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction import image as img_feature_extraction
from sklearn.externals import joblib
from StringIO import StringIO
from urlparse import urlparse


def process_directory(directory):
    '''Returns an array of feature vectors for all the image files in a
    directory (and all its subdirectories). Symbolic links are ignored.

    Args:
      directory (str): directory to process.

    Returns:
      list of list of float: a list of feature vectors.
    '''
    print("Processing images in %s" % directory)
    training = []




    image_limit = 10  # limit # of images to process to speed up testing




    image_count = len(list(os.walk(directory))[0][2])
    loop_counter = 0
    progress_bar = ProgressBar(widgets=[ETA(), ' ', Bar()], max_value=image_count, term_width=80).start()
    for root, _, files in os.walk(directory):
        for file_name in files:
            if loop_counter > image_limit:
                return training

            file_path = os.path.join(root, file_name)
            loop_counter += 1
            progress_bar.update(loop_counter)
            for img_feature in process_image_file(file_path):
                if img_feature:
                    training.append(img_feature)
    progress_bar.finish()
    print("\nDone.")
    return training


def process_image_file(image_path):
    '''Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)

        # Multiply the effectiveness of our training set by slightly modifying it
        # repeatedly in small ways
        yield process_image(image)
        # yield process_image(image.filter(ImageFilter.BLUR))
        # yield process_image(image.filter(ImageFilter.EDGE_ENHANCE))
        # yield process_image(image.filter(ImageFilter.EDGE_ENHANCE_MORE))
        # yield process_image(image.filter(ImageFilter.GaussianBlur))
        # yield process_image(image.filter(ImageFilter.SMOOTH))
        # yield process_image(image.transpose(Image.FLIP_LEFT_RIGHT))
        # yield process_image(image.transpose(Image.FLIP_TOP_BOTTOM))
    except IOError:
        yield None


def process_image_url(image_url):
    '''Given an image URL it returns its feature vector

    Args:
      image_url (str): url of the image to process.

    Returns:
      list of float: feature vector.

    Raises:
      Any exception raised by urllib2 requests.

      IOError: if the URL does not point to a valid file.
    '''
    parsed_url = urlparse(image_url)
    request = urllib2.Request(image_url)
    # set a User-Agent and Referer to work around servers that block a typical
    # user agents and hotlinking. Sorry, it's for science!
    request.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux ' \
            'x86_64; rv:31.0) Gecko/20100101 Firefox/31.0')
    request.add_header('Referrer', parsed_url.netloc)
    # Wrap network data in StringIO so that it looks like a file
    net_data = StringIO(urllib2.build_opener().open(request).read())
    image = Image.open(net_data)
    return process_image(image)


def process_image(image, blocks=4):
    '''Given a PIL Image object it returns its feature vector.

    Args:
      image (PIL.Image): image to process.
      blocks (int, optional): number of block to subdivide the RGB space into.

    Returns:
      list of float: feature vector if successful. None if the image is not
      RGB.
    '''
    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]
    # img_data = np.array(image.getdata())
    # patches = img_feature_extraction.extract_patches_2d(img_data, (16, 16))
    # import ipdb; ipdb.set_trace()
    # print(patches.shape)



def show_usage():
    '''Prints how to use this program
    '''
    print("Usage: %s [class A images directory] [class B images directory]" %
            sys.argv[0])
    sys.exit(1)


def train(training_path_a, training_path_b, print_metrics=True):
    '''Trains a classifier. training_path_a and training_path_b should be
    directory paths and each of them should not be a subdirectory of the other
    one. training_path_a and training_path_b are processed by
    process_directory().

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
      print_metrics  (boolean, optional): if True, print statistics about
        classifier performance.

    Returns:
      A classifier (sklearn.svm.SVC).
    '''
    if not os.path.isdir(training_path_a):
        raise IOError('%s is not a directory' % training_path_a)
    if not os.path.isdir(training_path_b):
        raise IOError('%s is not a directory' % training_path_b)
    training_a = process_directory(training_path_a)  # marked as a 1
    training_b = process_directory(training_path_b)  # marked as a 0
    # data contains all the training data (a list of feature vectors)
    data = training_a + training_b
    # target is the list of target classes for each feature vector: a '1' for
    # class A and '0' for class B
    target = [1] * len(training_a) + [0] * len(training_b)
    # split training data in a train set and a test set. The test set will
    # containt 20% of the total
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,
            target, test_size=0.20)
    # define the parameter search space
    parameters = {
        'kernel': ['linear', 'rbf'],
        'C': [1, 10, 100, 1000],
        'gamma': [0.01, 0.001, 0.0001]
    }
    # search for the best classifier within the search space and return it
    clf = grid_search.GridSearchCV(svm.SVC(), parameters, verbose=10).fit(x_train, y_train)
    classifier = clf.best_estimator_
    if print_metrics:
        print()
        print('Parameters:', clf.best_params_)
        print()
        print('Best classifier score')
        print(metrics.classification_report(y_test,
            classifier.predict(x_test)))
    return classifier


def get_classifier(training_path_a=None, training_path_b=None):
    classifier_file_name = os.path.join(os.path.dirname(__file__), 'lol_or_not_classifier.pkl')
    if os.path.exists(classifier_file_name):
        #print("Loading existing classifier @ %s" % classifier_file_name)
        classifier = joblib.load(classifier_file_name)
    else:
        assert training_path_a, "No existing classifier file found so training_path_a argument is required"
        print('Training classifier...')
        classifier = train(training_path_a, training_path_b)
        joblib.dump(classifier, classifier_file_name, compress=9)
    return classifier


def main(training_path_a, training_path_b):
    '''Main function. Trains a classifier and allows to use it on images
    downloaded from the Internet.

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
    '''
    classifier = get_classifier(training_path_a, training_path_b)

    while True:
        try:
            print("Input an image url (enter to exit): "),
            image_url = raw_input()
            if not image_url:
                break
            features = process_image_url(image_url)
            print(classifier.predict(features))
        except (KeyboardInterrupt, EOFError):
            break
        except:
            exception = sys.exc_info()[0]
            print(exception)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        show_usage()
    main(sys.argv[1], sys.argv[2])
