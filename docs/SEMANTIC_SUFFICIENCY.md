# Semantic features for matched annotation completeness

Fixed matched-v2 train/validation splits:482/98 examples. No gold annotation indices, construction names or verdict labels in model inputs. Frozen NLI per sentence/claim at256tokens gives contradiction/entailment/neutral probabilities. Features:mean3,max3,std3. Existing MiniLM model encodes claim and each sentence at384tokens; normalized cosine mean,max,min,std gives4features. Compare NLI9,embedding4,combined13. StandardScaler and balanced logistic regressionC1 fit only training; threshold.5,seed369. No validation hyperparameter selection or calibration.

Target remains inclusion of at least one complete published rationale. Same-split matched distractors can create semantic/topic cues; model pretraining exposure unknown. Dataset revision is adaptive and validation claims were observed earlier. Even good metrics do not establish independent real-world sufficiency, source authentication or safe deployment. Preserve all outcomes and all feature variants; do not pick a winner afterward and claim confirmation.

Input/code/protocol/model hashes and separate cache permit resuming unchanged runs only. No new model download. Run python scripts\run_semantic_sufficiency.py from D:\apv-rag-tesla369. Output artifacts/semantic_sufficiency_v1. No manuscript or GitHub push.
