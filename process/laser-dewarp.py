#!/usr/bin/python
# A dewarping tool that rectifies a document based on analysis of lasers.

import math, argparse, numpy, cv2, cv
from numpy import polynomial as P
from scipy import stats, integrate, interpolate

import lasers

version = '0.3'
debug = False

def dewarp(image, laser, side='odd', frame='single', threshold=40, factor=1.0,
           mask=None):
  lasermask = lasers.findLaserImage(laser, threshold, mask=mask)
  if debug:
    cv2.imwrite('tmp/laser.png', lasermask)
  top, bottom = lasers.extractLasers(lasermask,
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

# Based on http://users.iit.demokritos.gr/~bgat/3337a209.pdf
def warpModel(topLaser, bottomLaser, size, heightFactor):
  A, B = topLaser.getEdges()
  D, C = bottomLaser.getEdges()
  #print A, B, D, C
  AB = calculatePoly(topLaser.getCurve(), A, B)
  DC = calculatePoly(bottomLaser.getCurve(), D, C)
  if debug:
    cv2.imwrite('tmp/poly-top.png', topLaser.processImage(poly=AB, knots=AB.get_knots()))
    cv2.imwrite('tmp/poly-bottom.png', bottomLaser.processImage(poly=DC, knots=DC.get_knots()))
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
  #xbins = []
  #ybins = []
  #for x in xrange(left, right, 50):
    #xbins.append(x)
    #ybins.append(curve[x])
  
  binCount = (right - left)/20
  binned = stats.binned_statistic(xrange(0, right - left), curve[left:right],
                                  statistic='mean', bins=binCount)
  ybins = binned[0]
  xbins = binned[1][:-1]
  for i in xrange(len(xbins)):
    xbins[i] = xbins[i] + left + (right-left)/(binCount*2)
  
  #base = P.polynomial.polyfit(xbins, ybins, 7)
  #basePoly = P.polynomial.Polynomial(base)
  #xbins = range(0, len(curve))
  #ybins = curve
  return interpolate.InterpolatedUnivariateSpline(xbins, ybins, k=3, bbox=[0, len(curve)])
  #return basePoly

def calculateArc(base, left, right, sourceWidth, heightFactor):
  #adjustedheight = P.polynomial.polymul([heightFactor], base.coef)
  #prime = P.polynomial.polyder(adjustedheight)
  #squared = P.polynomial.polymul(prime, prime)
  #poly = P.polynomial.Polynomial(P.polynomial.polyadd([1], squared))
  prime = base.derivative()
  def intF(x):
    p = prime(x)*heightFactor
    return math.sqrt(p*p + 1)

  integralSum = 0
  arcCurve = []
  for x in xrange(0, left+1):
    arcCurve.append(0)
  for x in xrange(left+1, right):
    integralSum += integrate.romberg(intF, x-1, x, divmax=20)
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
  parser.add_argument('--callibration', dest='callibration_path',
                      default='callibration.png',
                      help='An image of the lasers on the background for callibration')
  options = parser.parse_args()

  debug = options.debug
  callibration = cv2.imread(options.callibration_path)
  angle = 180 - lasers.findLaserAngle(callibration)
  image = lasers.rotate(cv2.imread(options.image_path), angle)
  laser = lasers.rotate(cv2.imread(options.laser_path), angle)
  mask = None
  if options.mask_path is not None:
    mask = cv2.imread(options.mask_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  print 'angle: ', angle
  result = dewarp(image, laser, side=options.side, frame=options.frame,
                  threshold=options.laser_threshold,
                  factor=options.height_factor, mask=mask)
  cv2.imwrite(options.output_path, result)


#import cProfile
#cProfile.run('main()')
if __name__ == '__main__':
  main()

