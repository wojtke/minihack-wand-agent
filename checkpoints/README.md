# Shared checkpoint

`shared-best.npz` contains only six NumPy weight/bias arrays for the 463 → 128 → 128 → 11 Q-network. Loading disables pickle. The same weights and observation encoder are used on all four levels. `shared-best.json` records hashes, training steps and the original per-level evaluation summary.

The weights were exported without quantization from a previous shared Double DQN agent, trained for **3,670,016 environment actions** over multiple experimental stages. Those stages included shaped rewards and modified replay/exploration. This small public repository does **not** reproduce that training lineage. Its trainer provides a simpler starting point for new experiments.

The original final evaluation used 1,000 held-out seeds per level: **100% Easy, 99.8% Medium, 80.3% Hard, 0% Pro**. No training bonus was included in evaluation. These are seed-level results on fixed native terrain, not evidence of unseen-map generalization. The exported model and encoder were checked against the original implementation on complete native episodes.

A fresh export sanity check used 20 games per level, starting at seed 210,000,000: Easy 20/20, Medium 20/20, Hard 12/20, Pro 0/20. This small sample is recorded separately in [export-validation.json](export-validation.json). Export parity checked 5,332 transitions across 20 complete games, with identical features, actions and native rewards.

The four presentation clips use this agent. Easy, Medium and Hard show complete successful episodes; Pro shows the first 60 actions of an episode and does not show a win. Clips are accelerated and cropped around the player. Full-map illustrations are separate presentation-only renders; neither the policy nor the playable demo receives them.

Run a new evaluation with:

```sh
python -m wand_agent.evaluate --games 1000 --seed-start 200000000 --out runs/final-test.json
```

Keep this seed range reserved if you use it for a final test. The command writes new evidence; it is not a replay of the historical evaluation.
