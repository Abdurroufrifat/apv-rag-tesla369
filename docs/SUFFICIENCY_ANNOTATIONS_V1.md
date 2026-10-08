# Annotation-derived sufficiency dataset

Target: whether the supplied sentence set includes at least one complete published SciFact rationale set. It is a constructed annotation-coverage task, not universal semantic evidence sufficiency or historical authentication.

Source: training claims from the existing checksum-verified SciFact archive. Exclude entire training connected components touching previously observed development documents or canonical claim text. Group by shared cited/annotated documents and normalized identical claim text; all derived variants stay in the parent group. Climate data unused. Deterministic group hash determines train/validation membership before fitting.

Positive context includes all annotated decisive sentences plus up to two background sentences. Negative removes all decisive sentences, retaining background. An additional partial-removal negative is included only when no alternative annotated rationale remains complete. Claims without background sentences are skipped to reduce an empty-context shortcut. Negative contexts can still be semantically informative; the target remains annotated rationale completeness.

Builder checks every label against all alternative rationale sets, excludes development evidence documents, and checks group/document separation between splits. Source/code/data hashes recorded. No model training performed. Annotation indices and construction names must never enter model inputs. Compare with context-length and claim-only controls because removal patterns can expose construction cues. Report group-held-out performance on the constructed task; do not claim independent real-world sufficiency validation.

Files data/processed/sufficiency_annotations_v1/examples.json, parents.json, manifest.json. Next: train a model using claim/evidence text only with training-side grouped fitting and evaluate on the frozen validation groups. Preserve failure outcomes. No new human labels, manuscript or GitHub push.
