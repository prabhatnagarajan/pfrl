# Known differences
- we don't mask that same things
- we sample live
- We use DQN vs. PPO
- PPO checkpoint frequency
- show examples of frames
- mention that atari grand challenge dataset should be downloaded
- differences in how we select checkpointed synthetic demonstrations
- We add the L1 thresholding loss

## How to run
```
python train_dqn_trex.py --load-demos ~/pfn/pfrl/synth_demos/Breakout/ --demo-type <agc|synth>
```

Additional options:
- ` --load-trex`: Parameters of trained TREX network
- `--load-demos`: Location of demonstrations
- `--gpu`: GPU device. Pass -1 if no GPU is available.
