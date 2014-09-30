# This is the scantailor-specific code. Given two Laser curves, it
# will output an xml file that can be passed to ST on the command line
# for dewarping.

# Needs refactored before it can work as a standalone script.

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
