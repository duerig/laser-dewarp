import termios, fcntl, sys, os, time
import piggyphoto as pp

localPath = '/home/spreads/raw/book'

def wait():
    fd = sys.stdin.fileno()

    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)

    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

    try:
        while 1:
            try:
                c = sys.stdin.read(1)
                break
            except IOError: pass
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

def runProgram(command):
    print 'RUNNING: ' + command
    os.system(command)

cameras = []
pil = pp.portInfoList()
for name, path in pp.cameraList(autodetect=True).toList():
    pi = pil.get_info(pil.lookup_path(path))
    print "Connecting to camera {} at {}".format(name, path)
    cam = pp.camera(autoInit=False)
    cam.port_info = pi
    cam.init()
    cameras.append(cam)

if len(cameras) != 1:
    print "Required 1 camera. Not enough or too many."
    exit(1)

cam = cameras[0]
#print cam.config.main.capturesettings.shutterspeed2
config = cam.config
config['capturesettings']['shutterspeed2'].value = '1/5'
config['capturesettings']['f-number'].value = 'f/11'
config['capturesettings']['imagequality'].value = 'JPEG Fine'
config['imgsettings']['iso'].value = '100'
config['imgsettings']['whitebalance'].value = 'Tungsten'
config['imgsettings']['imagesize'].value = '6000x4000'
config['settings']['recordingmedia'].value = 'SDRAM'
cam.config = config

print config['settings']['recordingmedia'].value

runProgram('mkdir -p ' + localPath)
runProgram('./lasers-on')
runProgram('./lights-on')
runProgram('./guide-on')

counter = -2

while True:
    counterStr = '%03d' % counter
    if counter == -2:
        counterStr = 'background'
    elif counter == -1:
        counterStr = 'hands'
    elif counter == 0:
        config = cam.config
        config['capturesettings']['shutterspeed2'].value = '1/8'
        cam.config = config
    print config['settings']['recordingmedia'].value
    print 'Press a key to scan page ' + counterStr
    wait()
    
    print 'start at:'
    os.system('date')

    # Capture laser scan
    runProgram('./lights-off')
    runProgram('./lasers-on')
    runProgram('./guide-off')
    laserFolder, laserName = cam.capture_image()
    print 'laser at:'
    os.system('date')

    # Capture page scan
    runProgram('./lasers-off')
    runProgram('./guide-off')
    runProgram('./lights-on')
    scanFolder, scanName = cam.capture_image()
    print 'scan at:'
    os.system('date')

    cam.download_file(scanFolder, scanName, localPath + '/' + counterStr + '.jpg')
    cam.download_file(laserFolder, laserName, localPath + '/' + counterStr + '-laser.jpg')
    print 'download at:'
    os.system('date')

    # Prepare for next scan
    runProgram('./guide-on')
    runProgram('./lasers-on')
    counter += 1
