# Photo OCR

OCR: Optical Character Recognition

Pipeline:

1. Text detection
2. Character segmentation
3. Character classification

## Sliding window detection

To recognise something in an image, we would move a rectangle that we examine pixel by pixel throughout the image, searching for text or people or whatever we're recognising

We then apply an _expansion operator_ to join together recognised pixels that are close together by filling in the pixels between them, so we end up with boundary boxes around text, for example

1 dimensional sliding window can also be used in character segmentation to find where we divide a character from another

## Artificial data

In the case of _Character classification_ we could use characters from differents fonts pasted on top of random backgrounds

We can also take a training example and artificially distort it, so we get many examples out of one

Distortion can also be added to speech recognition examples by adding background sounds, for example

Distortions must be representative of things that can come up in the test set

You need a low bias classifier (incresa number of features or hidden units) before working on creating artificial data

See if you can get 10x more data, one way is with _Artificial data synthesis_. It might be faster to manually get more data and label it. Third option is to crowd source it

## Organising work on a pipeline

We need a metric for the overall system, like accuracy

### Ceiling analysis

We then go through each step of the pipeline, making it perfect for that metric. Like making the _Text detection_ step 100% accurate. We then chack the accuracy of the overall system, to see which step of the pipeline we should improve for better performance of the overall system


