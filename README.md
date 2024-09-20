# Exploring combos in Street Fighter Alpha 3 with Stable Retro

This is a project using AI to explore and identify the bests combos in Street Fighter Alpha 3, using Reinforcement Learning(RL)

## Description
The goal of this project is to find the most effective AI model for achieving the given objectives. The AI is rewarded for making the right decisions. Our primary aim in the article is for AI to identify the best combo based on time, but it can also be tested in different scenarios. For example, you can evaluate the AI's performance based on damage, damage + time, number of hits, or number of hits + time. Feel free to explore these scenarios and experiment with adjusting the parameters to improve results.

In our RL model, we accelerate training by using image transformations, frame stacking, and button filtering. These changes significantly enhance the AI's performance.

As well, we adapt the code from another work in literature using GA, in this [article](https://homepages.dcc.ufmg.br/~chaimo/public/Gecco16), to use as a comparison of our model, the results is described in the slide or article

> **Note:** we can't distribute the game's rom. In order for the code to work, you need to place a *Street Fighter Alpha 3 (USA) GBA* ROM inside `custom_integrations/StreetFighterAlpha3-GbAdvance` and name it `rom.gba`.

## Getting Started

### Dependencies

To run the code you will need use **Linux** and libraries in **"requirements.txt"**

### Executing program

1. Place the game's ROM inside `custom_integrations/StreetFighterAlpha3-GbAdvance` and name it `rom.gba`.
2. Run the `train.py` script to train the model using **A2C** or **PPO**.
   - You can run it without any arguments to use the default settings or customize the arguments as needed.

By default, it periodically saves the model to prevent loss of progress before overfitting occurs.

## Future works and improvements

The training of this AI and this ambient is very slow to reach a overfit, so give it a try to improve this things below

- Experiment with models using more timesteps to achieve overfitting.
- Explore additional methods to speed up training.
- Fine-tune hyperparameters for better performance.

## Authors

- **Rennam Victor Cabral de Faria**  
  [GitHub](https://github.com/RennamFaria)
- **Rodrigo Peixe Oliveira**  
  [GitHub](https://github.com/rpeixe)

## Main libraries used

- [Stable-Retro](https://github.com/Farama-Foundation/stable-retro)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
