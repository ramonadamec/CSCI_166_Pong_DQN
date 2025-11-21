# CSCI_166_Atari_DQN_Project
Deep Q-Learning on Atari Pong: Baseline DQN and Double DQN

This is the repo for our CSCI 166 project on Atari Pong with Deep Q Learning.
One such game we used was Pong, which we ran both as a normal DQN and as a Double DQN in our setup.

Folders in this repo:

- Baseline/
- DDQN/
- Starter/

Baseline
Plain DQN run.

- Baseline/CSCI_166_Project_Pong_(Baseline_DQN).ipynb
- Baseline/pong_dqn_returns.csv
- Baseline/pong_dqn_curve.png
- Baseline/pong_baseline_hparams.txt
- Baseline/pong_baseline_early.mp4
- Baseline/pong_baseline_learned.mp4

DDQN
Same game with targets computed using Double DQN.

- DDQN/CSCI_166_Project_Pong_(Double_DQN).ipynb
- DDQN/pong_ddqn_returns.csv
- DDQN/pong_ddqn_curve.png
- DDQN/pong_ddqn_hparams.txt
- DDQN/pong_ddqn_early.mp4
- DDQN/pong_ddqn_learned.mp4

Starter
We produced the course from the original notebook from the class.

- Starter/c166f25_02b_dqn_pong.ipynb

Game: ALE/Pong-v5

Short notes so we remember what we used:

- observations are the 4 stacked grayscale frames after preprocessing
- frames are resized to 84 x 84 resulting in the shape of (4, 84, 84)
- action space consists of 6 discrete actions from the ALE Pong environment
Reward is +1 when we score, -1 if the opponent scores and 0 otherwise, so rewards are pretty sparse

Both runs have the same configuration:

- same conv network but with a fully connected head on top
- same replay buffer size and same batch size
- same gamma, optimizer, and target network sync frequency
- same epsilon schedule (with a small adjustment when we adjusted training)

Main difference:

- Baseline DQN takes the maximum of the output of the target network for the next state.
- In Double DQN, the online net selects the next action, and the target net evaluates that action

Otherwise, they should converge the same as the original, except instead of a max target they use Double DQN.

What we did:

1. Open the provided Baseline/CSCI_166_Project_Pong_(Baseline_DQN).ipynb notebook on Google Colab (with the runtime type set to GPU) and run all of the cells in order.
2. Open DDQN/CSCI_166_Project_Pong_(Double_DQN).ipynb, enable GPU runtime, and run all code cells sequentially.

Each notebook will:

- install Atari and Gymnasium dependencies
- build the Pong environment with preprocessing
- train the agent
- write the *_returns.csv file
- save the *_curve.png learning curve
- save the early and learned *.mp4 clips for that run
