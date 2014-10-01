#!/usr/bin/python
# A dewarping tool that rectifies a document based on analysis of lasers.

import math, argparse, numpy, cv2, cv
from numpy import polynomial as P
from scipy import stats, integrate

version = '0.3'
debug = False

def dewarp(image, laser, side='odd', frame='single', threshold=40, factor=1.0,
           mask=None):
  lasermask = findLaserImage(laser, threshold, mask=mask)
  if debug:
    cv2.imwrite('tmp/laser.png', lasermask)
  top, bottom = extractLasers(lasermask,
                              side == 'odd' or side == 'right',
                              frame == 'single')

  if debug:
    cv2.imwrite('tmp/process-top.png', top.processImage())
    cv2.imwrite('tmp/process-bottom.png', bottom.processImage())

  model = warpModel(top, bottom, (image.shape[1], image.shape[0]), factor)
  result = dewarpFromModel(image, model)
  if mask is not None:
    dewarpedMask = dewarpFromModel(mask, model)
    cv2.add(result, cv2.merge((dewarpedMask, dewarpedMask, dewarpedMask)), result)
    inverted = numpy.subtract(255, dewarpedMask)
    topMin = inverted.shape[0]
    bottomMax = 0
    for x in xrange(0, inverted.shape[1]):
      column = numpy.nonzero(inverted[:, x])[0]
      if len(column) > 0:
        topMin = min(column[0], topMin)
        bottomMax = max(column[-1], bottomMax)
    result = result[topMin:bottomMax, :]
  return result

###############################################################################

def findLaserImage(image, threshold, mask=None):
  red = cv2.split(image)[2]
  if mask is not None:
    cv2.subtract(red, mask, red)
  retval, mask = cv2.threshold(red, threshold, 255, cv2.THRESH_BINARY)
  return cv2.medianBlur(mask, 5)

def extractLasers(image, isOdd, isSingle):
  contours, hierarchy = cv2.findContours(numpy.copy(image),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
  points = numpy.concatenate(contours)
  middleY = (numpy.amin(points[:,:,1]) + numpy.amax(points[:,:,1]))/2
  top = extractLaserPoints(image, (0, middleY))
  bottom = extractLaserPoints(image, (middleY, image.shape[0]))
  return (Laser(image, top, True, isOdd, isSingle),
          Laser(image, bottom, False, isOdd, isSingle))

def extractLaserPoints(image, yBound):
  result = []
  for x in xrange(image.shape[1]):
    column = numpy.nonzero(image[yBound[0]:yBound[1], x])[0]
    result.append(numpy.add(column, yBound[0]))
  return result

###############################################################################

class Laser:
  def __init__(self, mask, points, isTop, isOdd, isSingle):
    self.mask = mask
    self.isTop = isTop
    self.isOdd = isOdd
    self.isSingle = isSingle
    self.curve = []
    self.first = 0
    self.last = 0
    self.findCurve(points)
    self.spine = self.findSpine()
    self.edge = self.findEdge()

  def findCurve(self, points):
    self.curve, self.first, self.last = extractCurve(points)

  def findSpine(self):
    width = self.last - self.first
    start = self.first + int(width/3)
    end = self.first + int(2*width/3)
    if self.isSingle:
      if self.isOdd:
        end = start
        start = self.first
      else:
        start = end
        end = int(self.last)
    peak = None
    if self.isTop:
      peak = findPeaks(self.curve, start=start, end=end,
                       offsetX=20, offsetY=-5, compare=isGreater)
    else:
      peak = findPeaks(self.curve, start=start, end=end,
                       offsetX=20, offsetY=5, compare=isLess)
    if len(peak) == 0:
      raise BaseException('Could not find spine')
    return peak[int(len(peak)/2)]

  def findEdge(self):
    width = self.last - self.first
    start = self.first + int(width/3)
    end = self.first
    increment = -1
    deltaBack = 16
    deltaForward = 0
    if self.isOdd:
      start = self.first + int(2*width/3)
      end = self.last
      increment = 1
      deltaBack = 0
      deltaForward = 16
    prime = getDerivative(self.curve, deltaBack, deltaForward)
    peak = None
    if self.isTop:
      peak = findPeaks(prime, start=start, end=end,
                       increment=increment, offsetX=3, offsetY=0,
                       compare=isGreater)
    else:
      peak = findPeaks(prime, start=start, end=end,
                       increment=increment, offsetX=3, offsetY=0,
                       compare=isLess)
    return findFirstEdge(prime, peak, end)

  def getCurve(self):
    return self.curve

  def getEdges(self):
    if self.isOdd:
      return (self.spine, self.edge)
    else:
      return (self.edge, self.spine)

  def processImage(self, poly=None):
    left, right = self.getEdges()
    image = cv2.merge((self.mask, self.mask, self.mask))
    for i in xrange(left, right):
      image[self.curve[i], i] = (0, 0, 255)
      if poly is not None:
        image[poly(i), i] = (255, 255, 0)
    return image

###############################################################################

def extractCurve(points):
  first = None
  last = None
  curve = []
  previous = 0
  for x in xrange(0, len(points)):
    current = previous
    if len(points[x]) > 0:
      current = float(points[x][-1] + points[x][0]) / 2
      if first is None:
        first = x
      last = x
    curve.append(current)
    previous = current
  return curve, first, last

def findFirstEdge(points, candidates, default):
  result = default
  clipped, low, high = stats.sigmaclip(points, low=3.0, high=3.0)
  for candidate in candidates:
    if points[candidate] < low or points[candidate] > high:
      result = candidate
      break
  return result

def isGreater(a, b):
  return a >= b

def isLess(a, b):
  return a <= b

def getDerivative(curve, deltaBack, deltaForward):
  result = []
  for i in xrange(0, len(curve)):
    deltaLeft = max(i - deltaBack, 0)
    deltaRight = min(i + deltaForward, len(curve) - 1)
    result.append(float(curve[deltaRight] - curve[deltaLeft])/(deltaBack + deltaForward))
  return result

def findPeaks(points, start=0, end=0, increment=1,
              offsetX=1, offsetY=0, compare=isLess):
  results = []
  i = start
  while i != end:
    left = constrainPoint(i - offsetX, start, end)
    right = constrainPoint(i + offsetX, start, end)
    if (isPeak(points, candidate=i, end=left,
               increment=-1, compare=compare) and
        isPeak(points, candidate=i, end=right,
               increment=1, compare=compare) and
        taller(points[i], test=points[left],
               offset=offsetY, compare=compare) and
        taller(points[i], test=points[right],
               offset=offsetY, compare=compare)):
      results.append(i)
    i += increment
  return results

def constrainPoint(pos, start, end):
  result = pos
  if start < end:
    if pos < start:
      result = start
    if pos > end - 1:
      result = end - 1
  else:
    if pos > start:
      result = start
    if pos < end + 1:
      result = end + 1
  return result

def isPeak(points, candidate=0, end=1, increment=1, compare=isLess):
  result = True
  i = candidate
  while i != end:
    if not compare(points[candidate], points[i]):
      result = False
    i += increment
  return result

def taller(candidate, test=0, offset=0, compare=isLess):
  return compare(candidate + offset, test)

###############################################################################

# Based on http://users.iit.demokritos.gr/~bgat/3337a209.pdf
def warpModel(topLaser, bottomLaser, size, heightFactor):
  A, B = topLaser.getEdges()
  D, C = bottomLaser.getEdges()
  #print A, B, D, C
  AB = calculatePoly(topLaser.getCurve(), A, B)
  DC = calculatePoly(bottomLaser.getCurve(), D, C)
  if debug:
    cv2.imwrite('tmp/poly-top.png', topLaser.processImage(poly=AB))
    cv2.imwrite('tmp/poly-bottom.png', bottomLaser.processImage(poly=DC))
  ABarc = calculateArc(AB, A, B, size[0], heightFactor)
  DCarc = calculateArc(DC, D, C, size[0], heightFactor)
  width = max(ABarc[B], DCarc[C])
  height = min(distance([A, AB(A)], [D, DC(D)]),
               distance([B, AB(B)], [C, DC(C)]))
  startY = AB(A)
  finalWidth = int(math.ceil(width))

  map_x = numpy.asarray(cv.CreateMat(size[1], finalWidth, cv.CV_32FC1)[:,:])
  map_y = numpy.asarray(cv.CreateMat(size[1], finalWidth, cv.CV_32FC1)[:,:])

  topX = A
  bottomX = D
  for destX in xrange(A, finalWidth + A):
    Earc = (destX - A) / float(width) * ABarc[B]
    while topX < B and ABarc[topX] < Earc:
      topX += 1
    E = [topX, AB(topX)]
    while bottomX < C and DCarc[bottomX]/DCarc[C] < Earc/ABarc[B]:
      bottomX += 1
    G = [bottomX, DC(bottomX)]
    sourceAngle = math.atan2(G[1] - E[1], G[0] - E[0])
    cosAngle = math.cos(sourceAngle)
    sinAngle = math.sin(sourceAngle)
    distanceEG = distance(E, G) / height
    for destY in xrange(0, size[1]):
      sourceDist = (destY - startY) * distanceEG
      map_x[destY, int(destX - A)] = E[0] + sourceDist * cosAngle
      map_y[destY, int(destX - A)] = E[1] + sourceDist * sinAngle
  return (map_x, map_y)

def dewarpFromModel(source, model):
  dest = None
  if len(source.shape) >= 3:
    dest = numpy.zeros((model[0].shape[0], model[0].shape[1], source.shape[2]), dtype=source.dtype)
  else:
    dest = numpy.zeros((model[0].shape[0], model[0].shape[1]), dtype=source.dtype)
  cv2.remap(source, model[0], model[1], cv2.INTER_LINEAR, dest,
            cv2.BORDER_CONSTANT, (255, 255, 255))
  return dest

def calculatePoly(curve, left, right):
  binCount = (right - left)/50
  binned = stats.binned_statistic(xrange(0, right - left), curve[left:right],
                                  statistic='mean', bins=binCount)
  ybins = binned[0]
  xbins = binned[1][:-1]
  for i in xrange(len(xbins)):
    xbins[i] = xbins[i] + (right-left)/(binCount*2) + left
  base = P.polynomial.polyfit(xbins, ybins, 7)
  basePoly = P.polynomial.Polynomial(base)
  return basePoly

def calculateArc(base, left, right, sourceWidth, heightFactor):
  adjustedheight = P.polynomial.polymul([heightFactor], base.coef)
  prime = P.polynomial.polyder(adjustedheight)
  squared = P.polynomial.polymul(prime, prime)
  poly = P.polynomial.Polynomial(P.polynomial.polyadd([1], squared))
  def intF(x):
    return math.sqrt(poly(x))

  integralSum = 0
  arcCurve = []
  for x in xrange(0, left):
    arcCurve.append(0)
  for x in xrange(left, right):
    integralSum = integrate.romberg(intF, left, x, divmax=20)
    arcCurve.append(integralSum)
  for x in xrange(right, sourceWidth):
    arcCurve.append(integralSum)
  return arcCurve

def distance(a, b):
  return math.sqrt((a[0]-b[0])*(a[0]-b[0]) +
                   (a[1]-b[1])*(a[1]-b[1]))

###############################################################################

def main():
  global debug
  parser = argparse.ArgumentParser(
    description='A program for dewarping images based on laser measurements taken during scanning.')
  parser.add_argument('--version', action='version',
                      version='%(prog)s Version ' + version,
                      help='Get version information')
  parser.add_argument('--debug', dest='debug', default=False,
                      action='store_const', const=True,
                      help='Print extra debugging information and output pictures to ./tmp while processing (make sure this directory exists).')
  parser.add_argument('--image', dest='image_path', default='image.jpg',
                      help='An image of a document to dewarp')
  parser.add_argument('--laser', dest='laser_path', default='laser.jpg',
                      help='A picture with lasers on and lights out taken of the same page as the image.')
  parser.add_argument('--output', dest='output_path', default='output.png',
                      help='Destination path for dewarped image')
  parser.add_argument('--page', dest='side', default='odd',
                      help='Which side of the spine the page to dewarp is at. Can be either "odd" (equivalent to "right") or "even" (equivalent to "left")')
  parser.add_argument('--frame', dest='frame', default='single',
                      help='The number of pages in the camera shot. Either "single" if the camera is centered on just one page or "double" if the camera is centered on the spine')
  parser.add_argument('--laser-threshold', dest='laser_threshold',
                      type=int, default=40,
                      help='A threshold (0-255) for lasers when calculating warp. High means less reflected laser light will be counted.')
  parser.add_argument('--height-factor', dest='height_factor',
                      type=float, default=1.0,
                      help='The curve of the lasers will be multiplied by this factor to estimate height. The closer the lasers are to the center of the picture, the higher this number should be. When this number is too low, text will be foreshortened near the spine and when it is too high, the text will be elongated. It should normally be between 1.0 and 5.0.'),
  parser.add_argument('--mask', dest='mask_path', default=None,
                      help='A mask of the pages and book. The book contents should be black and any background, hands, or fingers should all be white.')
  options = parser.parse_args()

  debug = options.debug
  image = cv2.imread(options.image_path)
  laser = cv2.imread(options.laser_path)
  mask = None
  if options.mask_path is not None:
    mask = cv2.imread(options.mask_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  result = dewarp(image, laser, side=options.side, frame=options.frame,
                  threshold=options.laser_threshold,
                  factor=options.height_factor, mask=mask)
  cv2.imwrite(options.output_path, result)


#import cProfile
#cProfile.run('main()')
if __name__ == '__main__':
  main()

