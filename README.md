# Double Inverted Pendulum
Solving DIP(Double Inverted Pendulum) from OpenAI mujoco with SAC (Soft Actor-Critic)
The overall structure of the implementation of SAC was taken from the book "수학으로 풀어보는 강화학습 원리와 알고리즘."
However, the book dealt with a much simpler environment, and the code itself did not work when directly applied for DIP.
Therefore I had to study another source code (see folder), from which I realized the importance of the reparameterization trick.
The source code utilizes the original SAC structure (the old one with the value function); ny version of implementation rather utilizes the newer SAC structure (omitting the value function).



