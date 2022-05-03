# Reinforcement Learning: Puyo Puyo
Class project for CS394R: Reinforcement Learning

Xuyang Jin and Christopher Hahn


## Installation
Create a Python 3.6 environment using conda
```
conda create -n puyo python=3.6
conda activate puyo
```

Clone this repo
```
git clone https://github.com/WhyToFly/RL-proj.git
cd RL-proj
```

Install OpenMPI (following [OpenAI Spinning Up Installation Instructions](https://spinningup.openai.com/en/latest/user/installation.html))

Ubuntu:
```
sudo apt-get update && sudo apt-get install libopenmpi-dev
```

Mac OS:
```
brew install openmpi
```

Clone the [OpenAI Spinning Up Project](https://spinningup.openai.com/en/latest/index.html), install it
```
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
cd ..
```

Clone the [Puyo Puyo Gym Environment](https://github.com/frostburn/gym_puyopuyo), install it
```
git clone https://github.com/frostburn/gym_puyopuyo.git
cd gym_puyopuyo
pip install -e .
cd ..
```
Install [PyTorch](https://pytorch.org/)
```
conda install pytorch torchvision torchaudio -c pytorch
```
