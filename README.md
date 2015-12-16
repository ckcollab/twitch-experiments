twitch experiments
==================

Some things in here:

 * Grab frame from twitch stream
 * Find LoL character name from stream
 * OCR character name
 
setup
=====

```
brew install opencv tesseract
```
```
git clone git@github.com:ckcollab/twitch-experiments.git
```
```
cd twitch-experiments/src
```
```
pip install -r ../requirements.txt
```

And finally you may need to link to opencv in your virtualenv, note your openCV version number + virtual env name may be different
```
ln -s /usr/local/Cellar/opencv/2.4.12/lib/python2.7/site-packages/cv.py ~/.virtualenvs/twitch-experiments/lib/python2.7/site-packages
ln -s /usr/local/Cellar/opencv/2.4.12/lib/python2.7/site-packages/cv2.so ~/.virtualenvs/twitch-experiments/lib/python2.7/site-packages
```
