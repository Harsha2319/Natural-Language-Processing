# HMM Decoder - Viterbi Algorithm

## What the Code Does?
Computes probability of the test sentence using Viterbi Algorithm given the HMM Transition Probability and HMM Observation Likelihood.

## How to run?
### No Training
Bigram models in HMM Transition Probability.png and HMM Observation Likelihood.png are hardcoded in the program.

### Testing - Computes probability of test sentence
The program can be executed multiple times for testing once it is trained, But at each run, the program will only take one input sentence. The program takes 2 arguments (Program file name, test sentence)

  * Outputs: Maximum probability of the test sentence and the sequence of tags that resulted in the max probability. (Each word in the input sequence will be paired to the corresponding word in the output tag sequence)
  * Example: python probabilistic.py "closed in October and the jobs moved ." 

(NOTE: Test sentence must be included in double-quotes.)
