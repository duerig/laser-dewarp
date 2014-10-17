#!/usr/bin/python
# A dewarping tool that rectifies a document based on analysis of lasers.

import argparse, cv, cv2, math, numpy, os, re, sys
from scipy import stats, integrate, interpolate

import bookmask, handmodel, lasers

version = '0.4'
debug = False

def dewarp(image, laser, threshold=40, factor=1.0,
           mask=None):
  lasermask = lasers.findLaserImage(laser, threshold, mask=mask)
  if debug:
    cv2.imwrite('tmp/laser.png', lasermask)
  top, bottom = lasers.extractLasers(lasermask, True, True)

  if debug:
    cv2.imwrite('tmp/process-top.png', top.processImage())
    cv2.imwrite('tmp/process-bottom.png', bottom.processImage())

  model = warpModel(top, bottom, (image.shape[1], image.shape[0]), factor)
  result = dewarpFromModel(image, model)
  if mask is not None:
    dewarpedMask = dewarpFromModel(mask, model)
    cv2.add(result, cv2.merge((dewarpedMask, dewarpedMask, dewarpedMask)), result)
    inverted = numpy.subtract(255, dewarpedMask)
    topMax = None
    bottomMin = None
    for x in xrange(0, inverted.shape[1]):
      column = numpy.nonzero(inverted[:, x])[0]
      if len(column) > 0:
        if topMax is not None:
          topMax = max(column[0], topMax)
        else:
          topMax = column[0]
        if bottomMin is not None:
          bottomMin = min(column[-1], bottomMin)
        else:
          bottomMin = column[-1]
    result = result[topMax:bottomMin, :]
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
  ABarc = numpy.asarray(calculateArc(AB, A, B, size[0], heightFactor))
  DCarc = numpy.asarray(calculateArc(DC, D, C, size[0], heightFactor))
  width = max(ABarc[B], DCarc[C])
  height = min(distance([A, AB(A)], [D, DC(D)]),
               distance([B, AB(B)], [C, DC(C)]))
  startY = AB(A)
  finalWidth = int(math.ceil(width))

  map_x = numpy.asarray(cv.CreateMat(size[1], finalWidth, cv.CV_32FC1)[:,:])
  map_y = numpy.asarray(cv.CreateMat(size[1], finalWidth, cv.CV_32FC1)[:,:])

  topX = A
  bottomX = D
  destY = numpy.arange(0, size[1])

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

    sourceDist = (destY - startY) * distanceEG
    map_x[:, int(destX - A)] = E[0] + sourceDist * cosAngle
    map_y[:, int(destX - A)] = E[1] + sourceDist * sinAngle
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
  binCount = (right - left)/20
  binned = stats.binned_statistic(xrange(0, right - left), curve[left:right],
                                  statistic='mean', bins=binCount)
  ybins = binned[0]
  xbins = binned[1][:-1]
  for i in xrange(len(xbins)):
    xbins[i] = xbins[i] + left + (right-left)/(binCount*2)
  return interpolate.InterpolatedUnivariateSpline(xbins, ybins, k=3, bbox=[0, len(curve)])

def calculateArc(base, left, right, sourceWidth, heightFactor):
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

def findImages(root):
  result = []
  valid = re.compile('^[0-9]+.jpg$')
  all = os.listdir(root)
  for filename in all:
    path = os.path.join(root, filename)
    if (os.path.isfile(path) and
        valid.match(filename)):
      result.append(os.path.splitext(filename)[0])
  return result

def checkPath(name, path):
  if not os.path.exists(path):
    sys.stderr.write(name + ' not found: ' + path + '\n')
    exit(1)


def main():
  global debug
  parser = argparse.ArgumentParser(
    description='A program for dewarping images based on laser measurements taken during scanning. The input directory must have an image of just the background (background.jpg), just your hands over the background (hands.jpg), and the two horizontal lasers on the background (background-laser.jpg) and one or more scanned images.')
  parser.add_argument('--version', action='version',
                      version='%(prog)s Version ' + version,
                      help='Get version information')
  parser.add_argument('--debug', dest='debug', default=False,
                      action='store_const', const=True,
                      help='Print extra debugging information and output pictures to ./tmp while processing.')
  parser.add_argument('--output', dest='output_path', default='out',
                      help='Path where the resulting dewarped documents are stored. Defaults to ./out')
  parser.add_argument('--laser-threshold', dest='laser_threshold',
                      type=int, default=40,
                      help='A threshold (0-255) for lasers when calculating warp. High means less reflected laser light will be counted.')
  parser.add_argument('--stretch-factor', dest='stretch_factor',
                      type=float, default=1.0,
                      help='This parameter determines how much text will be stretched horizontally to remove foreshortening of the words. The stretching is concentrated near the spine of the book. When the lasers are far from the lens or the book is laid flat, it should be smaller. It is normally set between 1.0 and 5.0.')
  parser.add_argument('input_path',
                      help='Path where the documents to dewarp are stored. If this is a file, dewarp the single file. If this is a folder, dewarps all documents in the folder. The input directory must contain background.jpg, a hands.jpg, and a background-laser.jpg files and a "*-laser.jpg" file for every document to be dewarped.')
  options = parser.parse_args()
  debug = options.debug
  if debug:
    os.system('mkdir -p tmp')

  checkPath('input_path', options.input_path)
  checkPath('output_path', options.output_path)
  if os.path.isdir(options.input_path):
    basePath = options.input_path
    imageList = findImages(options.input_path)
  else:
    basePath, single_image = os.path.split(options.input_path)
    single_base, single_ext = os.path.splitext(single_image)
    if single_ext != '.jpg':
      sys.stderr.write('Input file must be a .jpg file: ' +
                       options.input_path + '\n')
      exit(1)
    imageList = [single_base]

  backgroundPath = os.path.join(basePath, 'background.jpg')
  checkPath('background path', backgroundPath)
  backgroundLaserPath = os.path.join(basePath, 'background-laser.jpg')
  checkPath('background-laser path', backgroundLaserPath)
  handPath = os.path.join(basePath, 'hands.jpg')
  checkPath('hands path', handPath)

  callibration = cv2.imread(backgroundLaserPath)
  angle = 180 - lasers.findLaserAngle(callibration)

  background = lasers.rotate(cv2.imread(backgroundPath), angle)
  hand = lasers.rotate(cv2.imread(handPath), angle)
  model = handmodel.create(background, [hand])

  for filename in imageList:
    imagePath = os.path.join(basePath, filename + '.jpg')
    checkPath('image path', imagePath)
    laserPath = os.path.join(basePath, filename + '-laser.jpg')
    checkPath('laser path', laserPath)
    sys.stderr.write('Dewarping ' + imagePath + '\n')

    image = lasers.rotate(cv2.imread(imagePath), angle)
    laser = lasers.rotate(cv2.imread(laserPath), angle)
    mask = bookmask.create(image, background, model)

    output = dewarp(image, laser, threshold=options.laser_threshold,
                    factor=options.stretch_factor, mask=mask)
    outputPath = os.path.join(options.output_path, filename + '.png')
    cv2.imwrite(outputPath, output)

#import cProfile
#cProfile.run('main()')
if __name__ == '__main__':
  main()

