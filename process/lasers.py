import cv2, math, numpy
from scipy import stats

def findLaserImage(image, background, threshold, mask=None):
  diff = cv2.subtract(image, background)
  #cv2.imwrite('diff.png', diff)
  channels = cv2.split(diff)
  red = channels[2]
  #red = cv2.addWeighted(channels[0], 1/3.0, channels[1], 1/3.0, 0)
  #red = cv2.addWeighted(red, 1.0, channels[2], 1.0, 0)
  cv2.imwrite('red.png', red)
  if mask is not None:
    cv2.subtract(red, mask, red)
  retval, mask = cv2.threshold(red, threshold, 255, cv2.THRESH_BINARY)
  result = mask
  result = cv2.medianBlur(mask, 7)
  #cv2.imwrite('laser-mask.png', result)
  return result

def extractSpines(image):
  start = image.shape[1]/3
  end = 2*start
  image = image[:,start:end]
  contours, hierarchy = cv2.findContours(numpy.copy(image),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
  points = numpy.concatenate(contours)
  middleY = (numpy.amin(points[:,:,1]) + numpy.amax(points[:,:,1]))/2
  top = extreme(image[0:middleY,:], numpy.argmax)
  bottom = extreme(image[middleY:image.shape[0],:], numpy.argmin)
  return ([top[0] + start, top[1]],
          [bottom[0] + start, bottom[1] + middleY])

def extreme(image, fun):
  contours, hierarchy = cv2.findContours(numpy.copy(image),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
  points = numpy.concatenate(contours)
  index = fun(points[:,:,1])
  return tuple(points[index][0])

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
    self.spine = 0
    self.edge = 0
    #self.spine = self.findSpine()
    #self.edge = self.findEdge()

  def findCurve(self, points):
    self.curve, self.first, self.last = extractCurve(points)

  def findSpine(self):
    width = self.last - self.first
    start = self.first + int(width/3)
    end = self.first + int(2*width/3)
    if self.isTop:
      self.spine = numpy.argmax(numpy.asarray(self.curve[start:end])) + start
    else:
      self.spine = numpy.argmin(numpy.asarray(self.curve[start:end])) + start
    #if self.isSingle:
    #  if self.isOdd:
    #    end = start
    #    start = self.first
    #  else:
    #    start = end
    #    end = int(self.last)
    #peak = None
    #if self.isTop:
    #  peak = findPeaks(self.curve, start=start, end=end,
    #                   offsetX=20, offsetY=-5, compare=isGreater)
    #else:
    #  peak = findPeaks(self.curve, start=start, end=end,
    #                   offsetX=20, offsetY=5, compare=isLess)
    #if len(peak) == 0:
    #  print 'Could not find spine'
    #  return (self.last + self.first) / 2
    #else:
    #  return peak[int(len(peak)/2)]

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
    #return end

  def getCurve(self):
    return self.curve

  def getEdges(self):
    return (self.first, self.last)
    #if self.isOdd:
      #return (int((self.first + self.last)/2), self.last)
      #return (self.spine, self.edge)
    #else:
      #return (self.first, int((self.first + self.last)/2))
      #return (self.edge, self.spine)

  def getAngle(self):
    rise = self.curve[0] - self.curve[-1]
    run = len(self.curve)
    return math.atan2(rise, run)*180/math.pi

  def processImage(self, poly=None, knots=None, bound=None):
    left, right = self.getEdges()
    if bound is not None:
      left, right = bound
    image = cv2.merge((self.mask, self.mask, self.mask))
    for i in xrange(left, right):
      image[self.curve[i], i] = (0, 0, 255)
      if poly is not None:
        image[int(poly(i)), i] = (255, 255, 0)
    if knots is not None:
      for knot in knots:
        for y in xrange(self.mask.shape[0]):
          if knot < self.mask.shape[1]:
            image[y, int(knot)] = (0, 255, 0)
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
  clipped, low, high = stats.sigmaclip(points, low=3.5, high=3.5)
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

def findLaserAngle(image, background, threshold=5):
  lasermask = findLaserImage(image, background, threshold)
  #cv2.imwrite('tmp/callibrate-laser.png', lasermask)
  top, bottom = extractLasers(lasermask, True, 'single')
  return top.getAngle()

def rotate(image, angle):
  matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2),
                                   angle, 1.0)
  return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

