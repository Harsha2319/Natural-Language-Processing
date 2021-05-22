# Smoothing Techniques

## What the Code Does?
Builds a bigram model based on the training data and Computes probability of the test sentence:
* without Smoothing
* with Add-One Smoothing
* with Good Turing Smoothing

## How to run?
### Training - Builds the Bigram model
If you run the code for the first time (ie. if the bigram models are not stores in the same folder as the program) with 1 arguments (Program file name) model will train on the training data and build the bigram model

Example : python smoothing.py

### Testing - Computes Probability of test sentence
Program can be executed multiple times for testing once it is trained, But at each run the program will only take one input sentence. During testing the progrem can take either 2 or 3 arguments

* 2 Arguments (Program file name, test sentence)
  * Outputs probabiliy of the test sentence using all 3 smoothing methods
  * Example : python smoothing.py "Brainpower , not physical plant now a firm"  

* 3 Arguments (Program file name, test sentence, smoothing method)
  * Outputs probability of the test sentence after appling the type of smoothing mentioned in the arguments
  * smoothing method argument can acceptes one of these values - noSmoothing, addOne, goodTuring.
  * Example : python smoothing.py "Brainpower , not physical plant now a firm" noSmoothing

(NOTE : Test sentence must be included in double quotes.)
