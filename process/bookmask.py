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
  background_mask = make_background_mask(source, background, hand_mask)
  return background_mask

def make_hand_mask(source, histogram):
  hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
  cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
  probability = cv2.calcBackProject([hsv], [0, 1], histogram,
                                    [0, 180, 0, 256], 1)
  cv2.erode(probability, disk, probability, (-1, -1), 2)
  cv2.imwrite('ghost.png', probability)
  retval, result = cv2.threshold(probability, 1, 255, cv2.THRESH_BINARY)
  cv2.imwrite('threshold.png', result)
  cv2.dilate(result, disk, result, (-1, -1), 6)
  mask = numpy.zeros((source.shape[0] + 4, source.shape[1] + 4), numpy.uint8)
  big = cv2.copyMakeBorder(result, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
  cv2.floodFill(big, mask, (0, 0), 128, 0, 0, cv2.FLOODFILL_FIXED_RANGE)
  result = big[1:-1, 1:-1]
  result = numpy.where(result == 0, 255, result)
  retval, result = cv2.threshold(result, 254, 255, cv2.THRESH_BINARY)
#  cv2.erode(result, disk, result, (-1, -1), 5)
#  cv2.dilate(result, disk, result, (-1, -1))
#  cv2.erode(result, disk, result, (-1, -1))
#  result = cv2.medianBlur(result, 9)
#  result = cv2.medianBlur(result, 45)
#  cv2.imwrite('blurred.png', result)
#  cv2.dilate(result, disk, result, (-1, -1), 5)
  cv2.imwrite('hand.png', result)
#  result = cut_hands(result)
#  cv2.imwrite('masked_hand.png', result)
  return cv2.merge((result, result, result))

def cut_hands(mask):
  contours, hierarchy = cv2.findContours(numpy.copy(mask),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    left = cnt[cnt[:,:,0].argmin()][0][0]
    right = cnt[cnt[:,:,0].argmax()][0][0]
    mask[0:-1,left:right] = 255
    print left, right
  return mask
#  largestContour = None
#  largestArea = 0.0
#  for contour in contours:
#    newArea = cv2.contourArea(contour)
#    if newArea > largestArea:
#      largestArea = newArea
#      largestContour = contour
#  rect = cv2.minAreaRect(largestContour)
#  print rect
#  cv2.drawContours(background_mask, (rect,), 0, 128)
#  return background_mask


def make_background_mask(source, background, hand_mask):
  size = source.shape
  cv2.subtract(source, hand_mask, source)
  cv2.subtract(source, background, source)
  cv2.imwrite('subtracted.png', source)
  mask = numpy.zeros((size[0] + 4, size[1] + 4), numpy.uint8)
  big = cv2.copyMakeBorder(source, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0, 0, 0))
  cv2.floodFill(big, mask, (0, 0), (255, 255, 255), (50,50,50), (5,5,5), cv2.FLOODFILL_FIXED_RANGE)
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
