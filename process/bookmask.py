#!/usr/bin/python

#
# bookmask
#
# Take an image of a book against a known background and hand model
# and create a mask of the book without either background or hands.
#

import argparse, cv2, numpy, handmodel

version = '0.1'

disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

def create(source, background, handImage):
  model = handmodel.create(background, [handImage])
  hand_mask = make_hand_mask(source, model)
  return make_background_mask(source, background, hand_mask)

def make_hand_mask(source, histogram):
  hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
  cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
  probability = cv2.calcBackProject([hsv], [0, 1], histogram,
                                    [0, 180, 0, 256], 1)
  median = cv2.medianBlur(probability, 25)
  for i in xrange(4):
    cv2.filter2D(median, -1, disk, median)
  retval, result = cv2.threshold(median, 5, 255, cv2.THRESH_BINARY)
  return cv2.merge((result, result, result))

def make_background_mask(source, background, hand_mask):
  size = source.shape
  cv2.subtract(source, hand_mask, source)
  cv2.subtract(source, background, source)
  big = numpy.array((size[0] + 2, size[1] + 2, 3), numpy.uint8)
  mask = numpy.zeros((size[0] + 4, size[1] + 4), numpy.uint8)
  big = cv2.copyMakeBorder(source, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0, 0, 0))
  cv2.floodFill(big, mask, (0, 0), (255, 255, 255), 100, 25)
  channels = cv2.split(big[1:-1, 1:-1])
  retval, blue = cv2.threshold(channels[0], 254, 255, cv2.THRESH_BINARY)
  retval, green = cv2.threshold(channels[1], 254, 255, cv2.THRESH_BINARY)
  retval, red = cv2.threshold(channels[2], 254, 255, cv2.THRESH_BINARY)
  result = cv2.bitwise_and(cv2.bitwise_and(blue, green), red)
  cv2.filter2D(result, -1, disk, result)
  return result

def main():
  parser = argparse.ArgumentParser(
      description='%(prog)s finds the book in an image, excluding the background and your hands or fingers. It must be callibrated with one or more background images and uses a hand model previously created with handmodel.py.')
  parser.add_argument('--version', action='version',
                      version='%(prog)s Version ' + version,
                      help='Get version information')
  parser.add_argument('--background', dest='background_path',
                      default='background.png',
                      help='Path to a background image. This image must be a photo taken of your background without any obtructions and under the same lighting conditions of your hands and book scans.')
  parser.add_argument('--hand', dest='hand_path',
                      default='hand.png',
                      help='An image with two hands in front of the background')
  parser.add_argument('input_path',
                      help='Path to a document image')
  parser.add_argument('output_path',
                      help='A mask image with the document piece black and the background and any hands or fingers in white')
  options = parser.parse_args()
  background = cv2.imread(options.background_path)
  #  hand = numpy.loadtxt(options.hand_path)
  hand = cv2.imread(options.hand_path)
  source = cv2.imread(options.input_path)
  result = create(source, background, hand)
  cv2.imwrite(options.output_path, result)

if __name__ == '__main__':
  main()
