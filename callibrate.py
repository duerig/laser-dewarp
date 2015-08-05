import os
from process import lasers
import cv2, numpy

threshold = 10
useGuide = False
useLasers = True

def init():
  os.system("gphoto2 --set-config /main/imgsettings/iso=0 " +
            "--set-config /main/capturesettings/shutterspeed2=32 " +
            "--set-config /main/imgsettings/whitebalance=3 " +
            "--set-config /main/capturesettings/f-number=10 " +
            "--set-config /main/imgsettings/imagesize=2 " +
            "--set-config /main/capturesettings/imagequality=0 ")

def captureGuide():
  os.system("gphoto2 --filename callibrate-guide.jpg " +
            "--capture-image-and-download --force-overwrite")

def captureLasers():
  os.system("gphoto2 --filename callibrate-lasers.jpg " +
            "--capture-image-and-download --force-overwrite")

def captureBackground():
  os.system("capture/all-off")
  #os.system("capture/lights-on")
  os.system("gphoto2 --filename callibrate-background.jpg " +
            "--capture-image-and-download --force-overwrite")

init()
captureBackground()
background = cv2.imread('callibrate-background.jpg')
background = cv2.resize(background, (500, 748))

while True:
  if useGuide:
    os.system("capture/lasers-off")
    os.system("capture/guide-on")
    captureGuide()
    guideImage = cv2.imread('callibrate-guide.jpg')
    guideImage = cv2.resize(guideImage, (500, 748))

  if useLasers:
    os.system("capture/guide-off")
    os.system("capture/lasers-on")
    captureLasers()
    laserImage = cv2.imread('callibrate-lasers.jpg')
    laserImage = cv2.resize(laserImage, (500, 748))

  if useGuide:
    guideImage = cv2.transpose(guideImage)
    cv2.flip(guideImage, 0, guideImage)
    guideMask = lasers.findLaserImage(guideImage, cv2.transpose(background), threshold=threshold)
    cv2.imwrite('guide-mask.png', guideMask)
    guidePoints = lasers.extractLaserPoints(guideMask, (0, guideImage.shape[0]))
    guideLaser = lasers.Laser(guideMask, guidePoints, True, True, True)
    print 'guide pos:', guideLaser.curve[0], '/', guideMask.shape[0]
    print 'guide:', - guideLaser.getAngle() - 90

  if useLasers:
    laserMask = lasers.findLaserImage(laserImage, background, threshold=threshold)
    cv2.imwrite('laser-mask.png', laserMask)
    topLaser, bottomLaser = lasers.extractLasers(laserMask, True, True)
    print 'top pos:', topLaser.curve[0], '/', laserImage.shape[0]
    print 'top:', topLaser.getAngle()
    print 'bottom pos:', bottomLaser.curve[0], '/', laserImage.shape[0]
    print 'bottom:', bottomLaser.getAngle()

  os.system("capture/guide-on")
  raw_input('Press enter...')
  
#os.system("capture/guide-on")
#os.system("capture/lasers-on")
