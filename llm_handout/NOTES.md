# NOTES

- From the experiments that I conducted, I found that the configuration showing the best perfomance apart fromt the given starter code was the one having byte level BPE tokenizer
- Experiment 1 showed that weight tying increased the BPE even though it decreased the number of parameters
- I also tried to fine tune our optimizer by replacing the existing Adam in starter with AdamW to take care of the weight decay. But we observed that it was in turn degrading the performance
- I then switched back to our original starter code and tried to work on the tokenizer later
- The largest improvement came from replacing the raw byte tokenizer with Byte level BPE
- The final submitted model uses the vocabulary size that achieved the lowerst BPP on the development set
- All experiments were conducted within the 2 million parameter limit and exactly 2000 optimizer steps.
- The tokenizer was trained only on the provided training corpus as required.
- The final submitted model uses the BPE vocabulary size that achieved the lowest BPB on the development set.
- The final checkpoint corresponds to the configuration with the best development BPB.
- The final submitted checkpoint (`ckpt.pt`) was verified using `evaluate.py` and achieved a BPB of 2.2238 with 1,509,440 parameters after 2000 training steps.

