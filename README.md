# Laser Book Scanning

DIY book scanning is about taking cheap consumer cameras and using
them to quickly and cheaply scan books. A flatbed scanner is
destructive and professionals use scanners that start at $10k and go
up from there.

Thus far, DIY book scanners have either reduced warping by physically
forcing the pages to be flat by usinga V-shaped platen made of glass
or plastic. Accomodating this platen complicates the design of the
scanner greatly, makes scanning small volumes hard, and slows down the
scanning process.

This project is about removing the need for a platen and using laser
beams to detect the shape of the page for later dewarping. The
laser-dewarp script handles two files for input. One is the normal
well-lit page being scanned. The other is a laser shot taken of the
same page moments before with the lights off and with laser lines
turned on.

# Requirements

To build a laser rig, you will need at least two line-focus lasers, a
camera, good lighting, and suitable hardware to mount all of these
things.

A Raspberry Pi is very useful to have as a way of automatically
managing the pieces. The lasers and lights will need to turn on and
off and the camera will need to take two pictures for every page
scanned. The Pi can act as the manager of all of these systems.

After you have scanned some images, you will need the following in
order to run the laser dewarping script:

* Python
* Numpy/Scipy
* Pillow

# Usage

<pre>
usage: laser-dewarp.py [-h] [--image IMAGE_PATH] [--laser LASER_PATH]
                       [--output OUTPUT_PATH] [--page SIDE]
                       [--spine-threshold SPINE_THRESHOLD]
                       [--laser-threshold LASER_THRESHOLD] [--bilinear]

A program for dewarping images based on laser measurements taken during
scanning.

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE_PATH    An image of a document to dewarp
  --laser LASER_PATH    A picture with lasers on and lights out taken of the
                        same page as the image.
  --output OUTPUT_PATH  Destination path for dewarped image
  --page SIDE           Which side of the spine the page to dewarp is at. Can
                        be either "odd" (equivalent to "right") or "even"
                        (equivalent to "left")
  --spine-threshold SPINE_THRESHOLD
                        A threshold (0-255) for lasers when detecting the
                        spine. Low means reflected laser light will be
                        counted.
  --laser-threshold LASER_THRESHOLD
                        A threshold (0-255) for lasers when calculating warp.
                        High means less reflected laser light will be counted.
  --bilinear            Use bilinear smoothing during dewarping which is
                        better but slower.
</pre>

# Discussion

This is very early days and nothing is certain yet. To discuss ideas
or ask questions, ask in the R&D section of the [DIY Book Scanner
Forum](http://www.diybookscanner.org/forum).

# Acknowledgements

The main author of the code is Jonathon Duerig, though I have taken
inspiration and ideas from Daniel Reetz, Christoph Nicolai (guitarguy)
and anonymous2 from the diybookscanner.org forums.
