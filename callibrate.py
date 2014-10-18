import os
from process import lasers
import cv2, numpy

def init():
  os.system("gphoto2 --set-config /main/imgsettings/iso=0 " +
            "--set-config /main/capturesettings/shutterspeed2=26 " +
            "--set-config /main/imgsettings/whitebalance=3 " +
            "--set-config /main/capturesettings/f-number=10 " +
            "--set-config /main/imgsettings/imagesize=0 " +
            "--set-config /main/capturesettings/imagequality=2 ")

def captureGuide():
  os.system("gphoto2 --filename callibrate-guide.jpg " +
            "--capture-preview --force-overwrite")

def captureLasers():
  os.system("gphoto2 --filename callibrate-lasers.jpg " +
            "--capture-preview --force-overwrite")

#def captureBackground():
#  os.system("capture/all-off")
#  os.system("capture/lights-on")
#  os.system("gphoto2 --filename callibrate-background.jpg " +
#            "--capture-image-and-download --force-overwrite")

init()
#captureBackground()

#os.system("capture/all-off")
#os.system("capture/guide-on")
#while True:
#  captureGuide()
#  guideImage = cv2.imread('callibrate-guide.jpg')
#  guideImage = cv2.transpose(guideImage)
#  cv2.imwrite('tmp/guide.png', guideImage)
#  cv2.flip(guideImage, 0, guideImage)
#  guideMask = lasers.findLaserImage(guideImage, threshold=10)
#  guidePoints = lasers.extractLaserPoints(guideMask, (0, guideImage.shape[0]))
#  guideLaser = lasers.Laser(guideMask, guidePoints, True, True, True)
#  print guideLaser.curve[0], guideLaser.curve[-1], guideMask.shape[1]
#  print 'guide:', - guideLaser.getAngle() - 90
#  raw_input('Press enter...')

os.system("capture/all-off")
os.system("capture/lasers-on")
while True:
  captureLasers()
  laserImage = cv2.imread('callibrate-lasers.jpg')
  laserMask = lasers.findLaserImage(laserImage, threshold=10)
  topLaser, bottomLaser = lasers.extractLasers(laserMask, True, True)

  print topLaser.curve[0], bottomLaser.curve[0], laserImage.shape[0]
  print 'top:', topLaser.getAngle()
  print 'bottom:', bottomLaser.getAngle()
  raw_input('Press enter...')
  
#os.system("capture/guide-on")
#os.system("capture/lasers-on")
