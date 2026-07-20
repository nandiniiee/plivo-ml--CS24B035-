# RUNLOG

## Experiment 0 (Baseline Experiment done at the start)
- None (Used the given starter code)
Results found: 
- Parameters: 1,339,840
- Final train loss: 1.7315
- BPB: 2.3718
- Time: 67 seconds

Observation: The given starter code provided us with these baseline results

## Experiment 1 (Weight tying = True instead of False)
- Changed tie_weights in model.py to True instead of given False
Results dound:
- Parameters: 1298880
- Final train loss: 1.7651
- BPB: 2.4122
- Time: 64 seconds

## Experiment 2 (Weight Tying = True and Optimizer now AdamW)
- Retained tie_weights=true and also changed the optimizer from Adam to AdamW to handle weight decay
- weight_decay=0.01 (chosen)
Results found:
- Parameters: 1298880
- Final train loss: 1.7975
- BPB: 2.4433
- Time: 68 seconds

## Experiment 3 (Weight tying and AdamW is hurting the model now)
- Experiment 1 and 2 both increased the BpB from the baseline. So instead of tuning optimizer or hyperparameters, we revert our model back to the original starter pack given
- Did a sanity check
- Results same as Baseline results

## Experiment 4 (From Byte Tokenizer, changed to Byte level tokenizer)
- Replaced the Byte Tokenizer with a trained byte level BPE tokenizer. Changed this in teh tokenizer.py file
- trained on train_corpus.txt
- We did this as the corpus mixes English and Hindi. And under pure byte level encoding, the Hindi text xonsumes fat more of the model's block size.
- changing it to BPE would let the model see more actual text
Results found:
- Parameters: 142176
- Final train loss: 2.9691
- BPB: 2.2369
- Time: 67 seconds

## Experiment 5 
