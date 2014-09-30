#!/usr/bin/python

#
# handmodel
#
# Generate a model for detecting hands in book scan images.
#
# Based on tutorials and suggestions from:
# http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_backprojection/py_histogram_backprojection.html#histogram-backprojection
# http://stackoverflow.com/questions/8593091/robust-hand-detection-via-computer-vision
#

import argparse, cv2, numpy

version = '0.1'

disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

def create(background, hands):
  histogram = None
  for image in hands:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = make_mask(background, image, hsv)
    if histogram:
      cv2.calcHist([hsv], [0, 1], mask, [180, 256],
                   [0, 180, 0, 256], histogram, True)
    else:
      histogram = cv2.calcHist([hsv], [0, 1], mask, [180, 256],
                               [0, 180, 0, 256])
  return histogram

def make_mask(background, foreground, hsv):
  withoutback = cv2.subtract(foreground, background)
  channels = cv2.split(withoutback)
  satchannel = cv2.split(hsv)[1]
  retval, blue = cv2.threshold(channels[0], 10, 255, cv2.THRESH_BINARY)
  retval, green = cv2.threshold(channels[1], 10, 255, cv2.THRESH_BINARY)
  retval, red = cv2.threshold(channels[2], 10, 255, cv2.THRESH_BINARY)
  retval, sat = cv2.threshold(satchannel, 20, 255, cv2.THRESH_BINARY)
  rgb = cv2.bitwise_or(cv2.bitwise_or(blue, green), red)
  base_mask = cv2.bitwise_and(rgb, sat)
  return cv2.erode(base_mask, disk)

def main():
  parser = argparse.ArgumentParser(
      description='%(prog)s accepts a background image and one or more photos of hands against that background to create a model for detecting hands when holding books.')
  parser.add_argument('--version', action='version',
                      version='%(prog)s Version ' + version,
                      help='Get version information')
  parser.add_argument('background_path',
                      help='Path to a background image. This image must be a photo taken of your background without any obtructions and under the same lighting conditions of your hands and book scans.')
  parser.add_argument('hands_path', nargs='+', metavar='hands_path',
                      help='The path to a photo of your hands against the background image. It must be under the same lighting conditions as the background image and image scans.')
  parser.add_argument('output_path',
                      help='Output path for hand model')
  options = parser.parse_args()

  background = cv2.imread(options.background_path)
  hands = []
  for path in options.hands_path:
    hands.append(cv2.imread(path))
  histogram = create(background, hands)
  print 'There is currently a bug in savetxt. Do not invoke this from the console for now. It is only for use as a library'
#  numpy.savetxt(options.output_path, histogram)

if __name__ == '__main__':
  main()
