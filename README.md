# Laser Book Scanning

DIY book scanning is about taking cheap consumer cameras and using
them to quickly and cheaply scan books. A flatbed scanner is
destructive and professionals use scanners that start at $10k and go
up from there. The goal of this project is to get a high quality
scanner that can be set up with only a couple of hundred dollars in
equipment.

Thus far, DIY book scanners have typically reduced warping by
physically forcing the pages to be flat using a V-shaped platen made
of glass or plastic. Accomodating this platen complicates the design
of the scanner greatly, makes scanning small volumes hard, and slows
down the scanning process.

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
* OpenCV (python-opencv)

# Usage

<pre>
usage: laser-dewarp.py [-h] [--version] [--debug] [--output OUTPUT_PATH]
                       [--upside-down] [--contrast CONTRAST]
                       [--brightness BRIGHTNESS] [--greyscale] [--grayscale]
                       [--deskew] [--laser-threshold LASER_THRESHOLD]
                       [--stretch-factor STRETCH_FACTOR]
                       input_path

A program for dewarping images based on laser measurements taken during
scanning. The input directory must have an image of just the background
(background.jpg), just your hands over the background (hands.jpg), and the two
horizontal lasers on the background (background-laser.jpg) and one or more
scanned images.

positional arguments:
  input_path            Path where the documents to dewarp are stored. If this
                        is a file, dewarp the single file. If this is a
                        folder, dewarps all documents in the folder. The input
                        directory must contain background.jpg, a hands.jpg,
                        and a background-laser.jpg files and a "*-laser.jpg"
                        file for every document to be dewarped.

optional arguments:
  -h, --help            show this help message and exit
  --version             Get version information
  --debug               Print extra debugging information and output pictures
                        to ./tmp while processing.
  --output OUTPUT_PATH  Path where the resulting dewarped documents are
                        stored. Defaults to ./out
  --upside-down         The source image is upside down, rotate 180 degrees
                        before processing
  --contrast CONTRAST   Adjust final image contrast (between 1.0 and 2.0)
  --brightness BRIGHTNESS
                        Adjust final image brightness (negative means darker)
  --greyscale           Output the resulting image in greyscale
  --grayscale           Output the resulting image in grayscale
  --deskew              Run a final content-based deskewing step before
                        outputting. This analyzes the text itself.
  --laser-threshold LASER_THRESHOLD
                        A threshold (0-255) for lasers when calculating warp.
                        High means less reflected laser light will be counted.
  --stretch-factor STRETCH_FACTOR
                        This parameter determines how much text will be
                        stretched horizontally to remove foreshortening of the
                        words. The stretching is concentrated near the spine
                        of the book. When the lasers are far from the lens or
                        the book is laid flat, it should be smaller. It is
                        normally set between 1.0 and 5.0.
</pre>

# Discussion

This is very early days and nothing is certain yet. To discuss ideas
or ask questions, ask in the R&D section of the [DIY Book Scanner
Forum](http://www.diybookscanner.org/forum).

# Acknowledgements

The main author of the code is Jonathon Duerig, though I have taken
inspiration and ideas from Daniel Reetz, Christoph Nicolai (guitarguy)
and anonymous2 from the diybookscanner.org forums.

# Changelog

Version 1.0:

* Added automatic deskewing based on spine position
* Cut each photo into two scans based on the spine
* Add a final deskew operation based on content after the spine for
  fine-tuning.
* Add brightness, contrast, and greyscale conversion options
* Add support for upside-down cameras, an option which turns everything
  around 180 degrees
* Laser dewarp postprocessing is now feature complete.

Version 0.3/0.4:

* Add background and hand detection. Autocrop both hands and the
  background. Use the hands to determine the left/right edges of the
  content and page.

Version 0.2:

* Various tweaks to the algorithm to be more faithful to the original paper.
* Added --height-factor option which can be increased to reduce
  foreshortening artifacts.

Version 0.1:

* Improved spine detection.
* Added page edge detection
* Removed the rotate-and-crop preprocessing step. This was interfering
  with the dewarping.
* Dewarping now uses the spine and page edge detection to define the
  sides of the source shape.
