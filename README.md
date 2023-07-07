## Various Pendulum Tasks
Here, various environments that deal with stabilizing or swinging-up pendulum(s) are solved with **RL** (Reinforcement Learning) approaches. Each folders
contain different *README.md* files, which you can read to find more about each subtask.

## Environments solved
Task | Env | Algorithm
| :---: | :---: | :---:
Single Pendulum Stabilization  | Cartpole-v1 | DDPG
Single Pendulum Swing-Up  | Pendulum-v1 | DDPG
Double Pendulum Swing-Up | Acrobot-v1 | SAC
Double Pendulum Stabilization | custom env | SAC
Double Pendulum Swing-Up & Stab.| custom env | SAC

*The first three environments are from the gym pacakge. They are all under the folder Practice_with_gym.*
*The double pendulum stabilization implementation was also succesfully tested on the mujoco environment!*

## Algorithm Used
* __DDPG__
* __SAC__ (seems way better, at least for pendulum-related tasks)

## NOTE
This repository was created just for myself to study reinforcement learning algorithms. No any other purposes, and I looked up many other open-source codes.
See the reference section above (or my code) if you also need some help implementing!

