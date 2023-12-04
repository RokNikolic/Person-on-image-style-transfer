## Putting a person on to an image while doing style transfer

Originally this project has been created to put a person onto a post card but also works for any arbitrary background.

### How it works

We are using [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/image_segmenter/) 
to cut a person out of their image. We use [TensorFlow](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)
to perform style transfer from the background image to the selfie. More specifically we are using
a [Magenta model](https://magenta.tensorflow.org/). Lastly we combine both images.

### Usage

- The program uses the first image in both "pictures of person" and "background" directories. So those must be present.
- We are using Python 3.11
- All the libraries are in the requirements.txt file