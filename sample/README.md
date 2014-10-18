# Sample Scans

These scans were made of a book in the public domain. In order to
generate dewarped versions from them run:

    process/laser-dewarp.py --output=. sample

You will probably notice quite a bit of foreshortening near the spine
on a couple of the scans which were held with the book partially
closed. If you want to experiment with the stretch factor to correct
this, you can try:

    process/laser-dewarp.py --stretch-factor=6.0 --output=. sample

