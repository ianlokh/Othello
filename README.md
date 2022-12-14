## TensorFlow (Metal) Installation

Please refer to <https://developer.apple.com/metal/tensorflow-plugin/> for the latest installation instructions.

For installation on MacOS with conda use miniforge instead of Anaconda

If you encounter the following error:
```commandline
ERROR: Could not find a version that satisfies the requirement tensorflow-macos (from versions: none)
ERROR: No matching distribution found for tensorflow-macos
```
you can try running the following to install tensorflow-macos
```
SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos tensorflow-metal
```
## Gym Installation

Run the following to install OpenAI Gym

```
pip install -U gym
pip install -U "gym[others]"
pip install -U "gym[atari]"
pip install -U "autorom[accept-rom-license]"
```


## Training the Reinforcement Learning agent 

To train the agent, run the command

```
python othello_train.py -m memory_profiler
```

Note that "-m memory_profiler" is optional and you can edit the code files to enable memory profiling


## Playing the game

To play the game, run the command


```
python othello_main.py
```

## Source Files

**othello_env.py**

This is the environment file for the Othello game.  This is subclassed from gym and overrides key functions such as:

```python

def render(self):
	...
	
def reset(self):
	...
	
def step(self):
	...
```

You can refer to <https://www.gymlibrary.dev/content/environment_creation/> for information on how to structure your environment class.


**othello_agent.py**

This is where we define the TensorFlow model and implementation of RL algorithms like DQN, greedy epsilon, replay buffer.


**othello_train.py**

This is where the traing of the RL agent happens.  There are many other training techniques but in this implementation, the agent is being trained against a player using random selection policy.

