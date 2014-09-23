#!/usr/bin/python
# A dewarping tool that rectifies a document based on analysis of lasers.

import os, sys, math, argparse, numpy, scipy
from PIL import Image, ImageMath, ImageFilter
from numpy import polynomial as P
from scipy import stats, integrate

options = None

class Line:
  def __init__(self, point1, point2):
    self.slope = (point2[1] - point1[1])/float(point2[0] - point1[0])
    self.point = point1

  def getY(self, x):
    return (x - self.point[0])*self.slope + self.point[1]

  def getX(self, y):
    return (y - self.point[1])/self.slope + self.point[0]

class Laser:
  def __init__(self, laserImage, yBound):
    self.curve = []
    self.spineIndex = 0
    self.findCurve(laserImage, yBound)

  def findCurve(self, laserImage, yBound):
    self.laserPoints = extractLaserPoints(laserImage, yBound)
    self.curve = extractCurve(self.laserPoints)

  def debugImage(self, laserImage, spine, laserColor, spineColor):
    for x in xrange(0, laserImage.size[0]):
      laserImage.putpixel((x, int(self.curve[x])), laserColor)
    for y in xrange(0, laserImage.size[1]):
      laserImage.putpixel((spine, y), spineColor)

#Numpy is a Python numerical library (matrices, etc.), so this just smooths data. This is basically all that I use numpy for.

def numpy_smooth(x, window_len = 11, window = 'hanning'):
  if window_len < 3:  return x
  
  if window == 'flat':
    w = numpy.ones(window_len, 'd')
  else:
    w = eval('numpy.' + window + '(window_len)')
  
  return numpy.convolve(w / w.sum(),
                        numpy.r_[2 * x[0] - x[window_len - 1::-1],
                                 x, 2 * x[-1] - x[-1:-window_len:-1]],
                        mode = 'same')[window_len: -window_len + 1]

###############################################################################

def findLaserImage(path, thresholdVal):
  def allOrNothing(x):
    if x > thresholdVal:
      return 255
    else:
      return 0
    
  image = Image.open(path)
  (channelR, channelG, channelB) = image.split()

  threshold = ImageMath.eval("convert(a-(b+c)/2, 'L')",
                                 a=channelR, b=channelG, c=channelB)
  threshold = Image.eval(threshold, allOrNothing)
  threshold = threshold.filter(ImageFilter.MedianFilter(5))
  return threshold

###############################################################################

def extractLaserPoints(image, yBound):
  pixels = image.load()
  # Loop over every column and add the y position of the laser points
  result = []
  for x in xrange(image.size[0]):
    column = []
    # Find all laser pixels in this column
    for y in xrange(yBound[0], yBound[1]):
      if pixels[x, y] > 200:
        column.append(y)
    result.append(column)
  return result

def extractCurve(points):
  curve = []
  lastPoint, lastIndex = findNextPoint(0, points, 0)
  for x in xrange(0, len(points)):
    nextPoint, nextIndex = findNextPoint(x, points, lastPoint)
    # If there are no remaining data points, just use the last data
    # point
    if nextIndex == -1 or nextIndex == lastIndex:
      curve.append(lastPoint)
    # Interpolate between the last data point and the next one. In
    # the degenerate case where nextIndex == x, this just results in
    # nextPoint.
    else:
#      totalWeight = abs(x - lastIndex) + abs(x - nextIndex)
#      last = abs(x - nextIndex) * lastPoint
#      next = abs(x - lastIndex) * nextPoint
#      curve.append((last + next) / float(totalWeight))
      curve.append(lastPoint)
    lastPoint = nextPoint
    lastIndex = nextIndex
  return curve#numpy_smooth(numpy.array(curve), window_len = 101)

def findNextPoint(start, points, last):
  resultPoint = last
  resultIndex = -1
  for i in xrange(start, len(points)):
    if len(points[i]) > 0:
      resultPoint = float(points[i][-1] + points[i][0]) / 2
      resultIndex = i
      break
  return (resultPoint, resultIndex)

def extractLasers(image):
  top = Laser(image, (0, image.size[1] / 2))
  bottom = Laser(image, (image.size[1] / 2, image.size[1]))
  return [top, bottom]

def extractSpines(curves):
  result = []
  for curve in curves:
    points = curve.laserPoints
    result.append(findExtreme(points, 0, len(points), 1, max))
  return result

#def extractSpine(points):
#  leftBulge = findExtreme(points, 0, len(points), 1, max)

#  seekLeft = seekOtherBulge(points, leftBulge, -1, -1)
#  seekRight = seekOtherBulge(points, leftBulge, len(points), 1)

#  rightBulge = seekRight
#  if len(points[seekLeft]) > len(points[seekRight]):
#    rightBulge = leftBulge
#    leftBulge = seekLeft

#  notch = findExtreme(points, leftBulge, rightBulge, 1, min)
#  return notch

#def seekOtherBulge(points, start, end, increment):
#  startThick = len(points[start])
#  done = False
#  i = start
#  while i != end and not done:
#    i += increment
#    if len(points[i]) < startThick*0.8:
#      done = True
#  return findExtreme(points, i, end, increment, max)

# ima is max (for thickest point) or min (for thinnest)
def findExtreme(points, start, end, increment, ima):
  extremeIndex = start
  if start != end and start >= 0:
    extreme = len(points[start])
    i = start
    while i != end:
      if ima(extreme, len(points[i])) != extreme:
        extreme = ima(extreme, len(points[i]))
        extremeIndex = i
      i += increment
  return extremeIndex

###############################################################################

def outputArcDewarp(imagePath, laserImage, spineImage, spines):
  source = Image.open(imagePath)
  curves = extractLasers(laserImage)
  makeProcessImage(laserImage, curves,
                   spines).save('tmp/process.png')
  unskewed = unskewImage(source, curves, spines, (255, 255, 255))
  unskewedLaser = unskewImage(laserImage, curves, spines, None)
  unskewedSpineImage = unskewImage(spineImage, curves, spines, None)
  unskewedCurves = extractLasers(unskewedSpineImage)
  unskewedSpines = extractSpines(unskewedCurves)
#  makeProcessImage(unskewedLaser, unskewedCurves,
#                   unskewedSpines).save('tmp/process-unskewed.png')
  if options.side == 'odd' or options.side == 'right':
    saveArcDewarp(unskewed, unskewedLaser, unskewedSpines, True)
  elif options.side == 'even' or options.side == 'left':
    saveArcDewarp(unskewed, unskewedLaser, unskewedSpines, False)

def saveArcDewarp(unskewed, unskewedLaser, spines, isOdd):
  suffix = '-even'
  if isOdd:
    suffix = '-odd'
  cropped = cropImage(unskewed, spines, isOdd)
#  cropped.save('tmp/cropped' + suffix + '.png')
  croppedLaser = cropImage(unskewedLaser, spines, isOdd)
  curves = extractLasers(croppedLaser);
#  makeProcessImage(croppedLaser, curves, []).save('tmp/process' + suffix + '.png')
  image = arcWarp(cropped, curves[0].curve, curves[1].curve)
  image.save(options.output_path)

def unskewImage(source, curves, spines, color):
  radians = math.atan2(curves[0].curve[spines[0]] - curves[1].curve[spines[1]],
                       spines[0] - spines[1])
  degrees = 90 + radians*180/math.pi
  if color:
    layer = source.convert('RGBA').rotate(degrees, Image.BILINEAR)
    result = Image.new('RGB', (layer.size[0], layer.size[1]), color)
    result.paste(layer, (0, 0), layer)
    return result
  else:
    return source.rotate(degrees, Image.BILINEAR)

def cropImage(source, spines, isOdd):
  leftX = max(spines[0], spines[1])
  if isOdd:
    leftX = min(spines[0], spines[1])
  cropped = None
  if isOdd:
    cropped = source.crop((leftX, 0, source.size[0], source.size[1]))
  else:
    cropped = source.crop((0, 0, leftX, source.size[1]))
  return cropped

###############################################################################

# Based on http://users.iit.demokritos.gr/~bgat/3337a209.pdf
def arcWarp(source, inAB, inDC):
  lastX = source.size[0] - 1
  AB = calculatePoly(inAB)
  DC = calculatePoly(inDC)
  ABarc = calculateArc(AB, len(inAB))
  DCarc = calculateArc(DC, len(inAB))
  width = max(ABarc[-1], DCarc[-1])
  height = max(DC(0) - AB(0),
               DC(lastX) - AB(lastX))
  startY = AB(0)
  finalWidth = int(math.ceil(width))
  dest = Image.new('RGB', [int(math.ceil(width)),
                           source.size[1]])
  canvas = dest.load()
  sourcePixels = source.load()

  topX = 0
  bottomX = 0
  for destX in xrange(0, finalWidth):
    Earc = destX / float(width) * ABarc[-1]
    while topX < lastX and ABarc[topX] < Earc:
      topX += 1
    E = [topX, AB(topX)]
    while bottomX < lastX and DCarc[bottomX]/DCarc[-1] < Earc/ABarc[-1]:
      bottomX += 1
    G = [bottomX, DC(bottomX)]
    sourceAngle = math.atan2(G[1] - E[1], G[0] - E[0])
    cosAngle = math.cos(sourceAngle)
    sinAngle = math.sin(sourceAngle)
    distanceEG = distance(E, G) / height
    for destY in xrange(0, source.size[1]):
      sourceDist = (destY - startY) * distanceEG
      sourceX = E[0] + sourceDist * cosAngle
      sourceY = E[1] + sourceDist * sinAngle
      if options.bilinear:
        canvas[destX, destY] = sampleSource(sourceX, sourceY, sourcePixels, source.size)
      else:
        canvas[destX, destY] = roundSource(sourceX, sourceY, sourcePixels, source.size)
  return dest

def calculatePoly(curve):
  binCount = len(curve)/50
  binned = stats.binned_statistic(xrange(0, len(curve)), curve,
                                  statistic='mean', bins=binCount)
  ybins = binned[0]
  xbins = binned[1][:-1]
  for i in xrange(len(xbins)):
    xbins[i] = xbins[i] + len(curve)/(binCount*2)
  base = P.polynomial.polyfit(xbins, ybins, 6)
  basePoly = P.polynomial.Polynomial(base)
  return basePoly

def calculateArc(base, width):
  prime = P.polynomial.polyder(base.coef)
  squared = P.polynomial.polymul(prime, prime)
  poly = P.polynomial.Polynomial(P.polynomial.polyadd([1], squared))
  def intF(x):
    return math.sqrt(poly(x))

  integralSum = 0
  arcCurve = [0]
  for x in xrange(1, width):
    piece = integrate.romberg(intF, x-1, x)
    integralSum += piece
    arcCurve.append(integralSum)
  return arcCurve

def distance(a, b):
  return math.sqrt((a[0]-b[0])*(a[0]-b[0]) +
                   (a[1]-b[1])*(a[1]-b[1]))

def roundSource(x, y, source, size):
  result = (255, 255, 255)
  intX = int(round(x))
  intY = int(round(y))
  if intX >= 0 and intY >= 0 and intX < size[0] and intY < size[1]:
    result = source[intX, intY]
  return result

def sampleSource(x, y, source, size):
  if x > 0 and y > 0 and x < size[0] - 1 and y < size[1] - 1:
    intX = int(x)
    intY = int(y)
    fracX = x - intX
    fracY = y - intY
    fracXY = fracX * fracY
    a = source[intX+1, intY+1]
    wa = fracXY
    b = source[intX+1, intY]
    wb = fracX - fracXY
    c = source[intX, intY+1]
    wc = fracY - fracXY
    d = source[intX, intY]
    wd = 1 - fracX - fracY + fracXY
    return (int(a[0]*wa + b[0]*wb + c[0]*wc + d[0]*wd),
            int(a[1]*wa + b[1]*wb + c[1]*wc + d[1]*wd),
            int(a[2]*wa + b[2]*wb + c[2]*wc + d[2]*wd))
  elif x >= 0 and y >= 0 and x < size[0] and y < size[1]:
    return source[x, y][0]
  else:
    return (255, 255, 255)

###############################################################################

def outputScanTailor(laserImage, spines):
  curves = extractLasers(laserImage)
#  processImage = makeProcessImage(laserImage, laserLines, spines)
#  processImage.save('tmp/scantailor-process.png')

  if options.odd_file:
    saveScanTailorParams(curves, spines, options.odd_file, True)
  if options.even_file:
    saveScanTailorParams(curves, spines, options.even_file, False)

def saveScanTailorParams(curves, spines, outPath, isOdd):
  stFile = open(outPath, 'w')
  stFile.write(scanTailorParams(curves[0], curves[1],
                                spines[0], spines[1], isOdd))
  stFile.close()
def scanTailorParams(top, bottom, topSpine, bottomSpine):
  spineSkew = topSpine - bottomSpine
  result = '<distortion-model>\n'
  result += scanTailorCurve('top-curve', top, topSpine, spineSkew)
  result += scanTailorCurve('bottom-curve', bottom, bottomSpine, -spineSkew)
  result += '</distortion-model>\n'
  return result

def scanTailorCurve(name, laser, spineIndex, spineSkew):
  last = min(len(laser.curve), len(laser.curve) + spineSkew) - 1
  result = '  <' + name + '>\n'
  result += '    <xspline>\n'
  for x in xrange(spineIndex, last, 50):
    result += scanTailorPoint(x, laser)
  if last % 50 != 49:
    result += scanTailorPoint(last, laser)
  result += '    </xspline>\n'
  result += '    <polyline></polyline>\n'
  result += '  </' + name + '>\n'
  return result

def scanTailorPoint(x, laser):
  return '      <point x="' + str(x) + '" y="' + str(laser.curve[x]) + '"/>\n'

###############################################################################

def makeProcessImage(source, curves, spines):
  result = Image.new('RGB', source.size)
  result.paste(source, (0, 0, source.size[0], source.size[1]))
  for i in xrange(0, len(curves)):
    spine = 0
    if i < len(spines):
      spine = spines[i]
    curves[i].debugImage(result, spine, (255, 255, 0), (0, 0, 255))
  return result

###############################################################################

def parseArgs():
  global options
  parser = argparse.ArgumentParser(
    description='A program for dewarping images based on laser measurements taken during scanning.')
  parser.add_argument('--image', dest='image_path', default='image.jpg',
                      help='An image of a document to dewarp')
  parser.add_argument('--laser', dest='laser_path', default='laser.jpg',
                      help='A picture with lasers on and lights out taken of the same page as the image.')
  parser.add_argument('--output', dest='output_path', default='output.png',
                      help='Destination path for dewarped image')
  parser.add_argument('--page', dest='side', default='odd',
                      help='Which side of the spine the page to dewarp is at. Can be either "odd" (equivalent to "right") or "even" (equivalent to "left")')
  parser.add_argument('--spine-threshold', dest='spine_threshold',
                      type=int, default=30,
                      help='A threshold (0-255) for lasers when detecting the spine. Low means reflected laser light will be counted.')
  parser.add_argument('--laser-threshold', dest='laser_threshold',
                      type=int, default=100,
                      help='A threshold (0-255) for lasers when calculating warp. High means less reflected laser light will be counted.')
  parser.add_argument('--bilinear', dest='bilinear', default=False,
                      action='store_const', const=True,
                      help='Use bilinear smoothing during dewarping ' +
                      'which is better but slower. Only affects arc-dewarp.')
  options = parser.parse_args()

###############################################################################

def main():
  parseArgs()

  spineImage = findLaserImage(options.laser_path, options.spine_threshold)
#  spineImage.save('tmp/spine-laser.png')
  spineLines = extractLasers(spineImage)
  spines = extractSpines(spineLines)

  laserImage = findLaserImage(options.laser_path, options.laser_threshold)
#  laserImage.save('tmp/laser.png')
  outputArcDewarp(options.image_path, laserImage, spineImage, spines)

#import cProfile
#cProfile.run('main()')
if __name__ == '__main__':
  main()

