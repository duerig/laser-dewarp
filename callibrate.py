import os
from process import lasers
import cv2

def init():
  os.system("gphoto2 --set-config /main/imgsettings/iso=0 " +
            "--set-config /main/capturesettings/shutterspeed2=26 " +
            "--set-config /main/imgsettings/whitebalance=3 " +
            "--set-config /main/capturesettings/f-number=10 " +
            "--set-config /main/imgsettings/imagesize=0 " +
            "--set-config /main/capturesettings/imagequality=2 ")

def captureGuide():
  os.system("capture/all-off")
  os.system("capture/guide-on")
  os.system("gphoto2 --filename callibrate-guide.jpg " +
            "--capture-image-and-download --force-overwrite")

def captureLasers():
  os.system("capture/all-off")
  os.system("capture/lasers-on")
  os.system("gphoto2 --filename callibrate-lasers.jpg " +
            "--capture-image-and-download --force-overwrite")

#def captureBackground():
#  os.system("capture/all-off")
#  os.system("capture/lights-on")
#  os.system("gphoto2 --filename callibrate-background.jpg " +
#            "--capture-image-and-download --force-overwrite")

init()
captureGuide()
captureLasers()
#captureBackground()

guideImage = cv2.imread('callibrate-guide.jpg')
cv2.transpose(guideImage, guideImage)
cv2.flip(guideImage, 0, guideImage)
guideMask = lasers.findLaserImage(guideImage, threshold=1)
guidePoints = lasers.extractLaserPoints(guideMask, (0, guideImage.shape[0]))
guideLaser = lasers.Laser(guideMask, guidePoints, True, True, True)

print 'guide:', - guideLaser.getAngle() - 90

laserImage = cv2.imread('callibrate-lasers.jpg')
laserMask = lasers.findLaserImage(laserImage, threshold=1)
topLaser, bottomLaser = lasers.extractLasers(laserMask, True, True)

print 'top:', topLaser.getAngle()
print 'bottom:', bottomLaser.getAngle()

os.system("capture/guide-on")
os.system("capture/lasers-on")
