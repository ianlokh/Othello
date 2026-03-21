import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from scipy.special import softmax

from othello import config as cfg
from othello.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer

# for performance profiling
# import cProfile as cprofile
# from memory_profiler import profile
# fp = open("report-agent.log", "w+")  # to capture memory profile logs

# Classic Othello positional weight matrix (from white's perspective).
# Corners (120) are extremely valuable — they can never be flipped.
# X-squares (-40, diagonally adjacent to corners) are dangerous to occupy.
# C-squares (-20, edge positions adjacent to corners) are also risky.
# fmt: off
POSITION_WEIGHTS = np.array([                                          # noqa: E201
    [120, -20,  20,   5,   5,  20, -20, 120],                         # noqa: E201
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],                         # noqa: E201
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],                         # noqa: E201
    [  5,  -5,   3,   3,   3,   3,  -5,   5],                         # noqa: E201
    [  5,  -5,   3,   3,   3,   3,  -5,   5],                         # noqa: E201
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],                         # noqa: E201
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],                         # noqa: E201
    [120, -20,  20,   5,   5,  20, -20, 120],                         # noqa: E201
], dtype=np.float64).flatten()  # shape (64,) to match observation layout
# fmt: on

'''
import tensorflow as tf
from tensorflow.keras import layers, models

class OthelloDQNModel:
    """
    Class for the deep neural network model with residual blocks.
    """
    def __init__(self, nb_observations, action_dim, learning_rate):
        self.nb_observations = nb_observations
        self.action_dim = action_dim
        self.learning_rate = learning_rate

    def residual_block(self, inputs, filters):
        """
        Define a residual block with two convolutional layers.

        :param inputs: Input tensor to the block.
        :param filters: Number of filters for the convolutions.
        :return: Output tensor after applying the residual block.
        """
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters, (3, 3), padding='same')(x)  # No activation here
        x = layers.BatchNormalization()(x)

        x = layers.Add()([x, inputs])  # Skip connection

        return layers.Activation('relu')(x)

    def build_model(self):
        """
        Build TensorFlow model with residual blocks.

        :return: TensorFlow model.
        """
        inputs = layers.Input(shape=(self.nb_observations, self.nb_observations, 1))

        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)

        # Apply a few residual blocks
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 64)

        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Flatten the output and add dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)

        outputs = layers.Dense(self.action_dim, activation=tf.keras.activations.linear)(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])

        return model
'''


class OthelloDQNModel(nn.Module):
    """
    PyTorch deep neural network model — exact port of the TensorFlow Sequential architecture.

    Architecture (mirrors TF build_model):
        Linear(64→64) + ReLU
        Linear(64→64) + ReLU
        Linear(64→64) + BatchNorm1d(64) + LeakyReLU       [block 1]
        Linear(64→128) + ReLU
        Linear(128→128) + ReLU
        Linear(128→128) + ReLU
        Linear(128→64) + BatchNorm1d(64) + LeakyReLU      [block 2]
        Linear(64→64) + ReLU
        Linear(64→action_dim)                              [linear output]
    """

    def __init__(self, nb_observations, action_dim):
        super().__init__()
        self.nb_observations = nb_observations
        self.action_dim = action_dim

        self.net = nn.Sequential(
            # Dense(64, relu) — input (*, 64)
            nn.Linear(nb_observations, 64),
            nn.ReLU(),
            # Dense(64, relu)
            nn.Linear(64, 64),
            nn.ReLU(),
            # Dense(64) + BatchNorm + LeakyReLU
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            # Dense(128, relu)
            nn.Linear(64, 128),
            nn.ReLU(),
            # Dense(128, relu)
            nn.Linear(128, 128),
            nn.ReLU(),
            # Dense(128, relu)
            nn.Linear(128, 128),
            nn.ReLU(),
            # Dense(64) + BatchNorm + LeakyReLU
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            # Dense(64, relu)
            nn.Linear(64, 64),
            nn.ReLU(),
            # Dense(action_dim, linear) — output
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        """
        Forward pass. Accepts both 2-D (batch, 64) and 3-D (batch, 1, 64) inputs.

        :param x: input tensor
        :return: Q-value tensor of shape (batch, action_dim)
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # (batch, 1, 64) → (batch, 64)
        return self.net(x)


class OthelloDQN:
    """
    Class for the OthelloDQN object
    """

    def __init__(self, nb_observations, player="white", mode="training", seed=42):

        self.set_global_determinism(seed=seed)

        self.mode = mode
        self.player = player

        self.action_dim = 64
        self.state_dim = 64

        self.gamma = cfg.agent_setting.GAMMA  # reward decay rate
        self.alpha1 = cfg.agent_setting.ALPHA1  # soft copy weights for self-play, alpha1 updates while (1-alpha1) remains
        self.alpha2 = cfg.agent_setting.ALPHA2  # soft copy weights from eval net to target net, alpha2 updates while (1-alpha2) remains
        self.epsilon_reduce = cfg.agent_setting.EPSILON_REDUCE  # 0.995, 0.9995, 0.99975, 0.9999, 0.999975

        if self.mode == "training":
            self.epsilon = cfg.agent_setting.EPSILON  # epsilon parameter for epsilon greedy selection training
        else:
            self.epsilon = 0  # will not train

        # q network learning parameters
        self.learning_rate = cfg.agent_setting.LEARNING_RATE  # 0.001, 0.0005, 0.0001
        self.batch_size = cfg.agent_setting.BATCH_SIZE  # 128, 256, 512, 768, 1024, 2048

        # total learning step - count how many times the eval net has been updated, used to set a basis for updating
        # the target net
        self.learn_step_counter = cfg.agent_setting.LEARN_STEP_COUNTER
        self.replace_target_iter = cfg.agent_setting.REPLACE_TARGET_ITER  # 10, 50, 75, 100, 150

        # replay buffer settings
        self.replay_buffer_size = cfg.agent_setting.REPLAY_BUFFER_SIZE
        if cfg.agent_setting.REPLAY_BUFFER_STRATEGY == "per":
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.replay_buffer_size,
                alpha=cfg.agent_setting.PER_ALPHA,
                beta=cfg.agent_setting.PER_BETA,
                beta_increment=cfg.agent_setting.PER_BETA_INCREMENT,
                epsilon=cfg.agent_setting.PER_EPSILON,
                abs_err_upper=cfg.agent_setting.PER_ABS_ERR_UPPER,
            )
        else:
            self.replay_buffer = UniformReplayBuffer(capacity=self.replay_buffer_size)

        # specify the q network path
        self.model_full_path = "./models/"

        # select compute device (MPS → CUDA → CPU)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )

        # only white player learns hence q network will only be created for white player
        if self.player == "white":
            self.model_eval = OthelloDQNModel(nb_observations, self.action_dim).to(self.device)
            self.model_eval.train()

            # optimizer: Adam with grad clipping (mirrors TF Adam(clipnorm=0.5))
            self.optimizer = torch.optim.Adam(self.model_eval.parameters(), lr=self.learning_rate)
            # LR schedule: decay by 0.9 every 10,000 steps (mirrors TF ExponentialDecay staircase=True)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10000, gamma=0.9
            )
            self.criterion = nn.MSELoss()

        # regardless of training (random or self-play) target network will always be created because this is the network
        # that will be used to predict the action
        self.model_target = OthelloDQNModel(nb_observations, self.action_dim).to(self.device)
        self.model_target.eval()

        # array to store the moves made by the agent
        self.epsilon_plays = []

        # performance profiling
        # self.cprof = cprofile.Profile()

    @staticmethod
    def set_env_seeds(seed):
        """
        sets the seed value so that we can reproduce the results constantly
        :param seed:
        :return:
        """
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        print("Seed:{:d}".format(seed))

    def set_global_determinism(self, seed):
        """
        sets pytorch specific deterministic parameters for reproducibility
        :param seed:
        :return:
        """
        self.set_env_seeds(seed=seed)

    @staticmethod
    def _compute_potential(observation):
        """Compute positional potential Phi(s) for PBRS reward shaping.

        Uses dot product of the flattened board state with POSITION_WEIGHTS.
        Positive values favour white, negative favour black (matches board encoding).

        :param observation: numpy array, shape (1, 64)
        :return: scalar potential value
        """
        return np.dot(POSITION_WEIGHTS, observation.flatten())

    # @profile(stream=fp)
    def store_transition(self, observation, action, reward, done, next_observation):
        """
        Store an experience transition into the replay buffer (white player only).

        :param observation: current state
        :param action: action taken
        :param reward: reward received
        :param done: whether the episode ended
        :param next_observation: resulting state
        :return: None
        """
        if self.player == "white":
            # Potential-based reward shaping (Ng et al. 1999):
            # F(s, s') = gamma * Phi(s') - Phi(s)
            # Preserves optimal policy while providing intermediate signal.
            if cfg.agent_setting.REWARD_SHAPING_ENABLED and not done:
                phi_s = self._compute_potential(observation)
                phi_s_prime = self._compute_potential(next_observation)
                shaping = self.gamma * phi_s_prime - phi_s
                reward = reward + cfg.agent_setting.REWARD_POSITIONAL_WEIGHT * shaping

            self.replay_buffer.add((observation, action, reward, next_observation, done))

    # @profile(stream=fp)
    def choose_action(self, observation, possible_actions):
        """
        This is an implementation of epsilon_greedy_action_selection to balance between exploitation and exploration
        :param observation: list[list], shape=[8, 8]
        :param possible_actions: a set of tuples (row, col)
        :return: a tuple of (row, col)
        """
        # performance profiling
        # self.cprof.enable()

        # set the mask
        mask = np.array([[True] * 64], dtype=bool)  # shape = (1, 64)
        for row, col in possible_actions:
            mask[0][(row * 8) + col] = False  # do not mask a possible action

        if np.random.random() > self.epsilon:
            observation = np.expand_dims(observation, axis=0)  # (1, 1, 64)
            obs_t = torch.FloatTensor(observation).to(self.device)

            with torch.no_grad():
                prediction = self.model_target(obs_t).cpu().numpy()  # (1, 64)

            prediction = softmax(np.ma.array(prediction, mask=mask).filled(fill_value=-1e9), axis=None)

            action = int(np.argmax(prediction[0]))

            # add the position the agent as decided on
            self.epsilon_plays.append(action)
        else:
            action = random.choice(list(possible_actions))
            action = (action[0] * 8) + action[1]

        # performance profiling
        # self.cprof.disable()
        return action

    @staticmethod
    def _soft_update(target_model, source_model, alpha):
        """Polyak (soft) update: target <- (1-alpha)*target + alpha*source."""
        target_sd = target_model.state_dict()
        source_sd = source_model.state_dict()
        new_sd = {}
        for key in target_sd:
            if target_sd[key].dtype.is_floating_point:
                new_sd[key] = target_sd[key] * (1 - alpha) + source_sd[key] * alpha
            else:
                new_sd[key] = source_sd[key].clone()
        target_model.load_state_dict(new_sd)

    # assign weights from trained agent into self-play agent
    # @profile(stream=fp)
    def assign_weights(self, other: "OthelloDQN"):
        """
        accept weights from the other (white) player where the weights from the trained agent will be copied and this
        agent will be used for self-play training
        :param other: trained agent from which the weights are to be copied from
        :return:
        """
        if self.player == "other":
            self._soft_update(self.model_target, other.model_eval, self.alpha1)
            print('Update weights from another agent for self-play')

    # sync between model_eval and model_target
    # @profile(stream=fp)
    def _tgt_evl_sync(self):
        """
        copies the weights from model_eval (Q network) to model_target (Target network). partial copy the weights from
        eval net to target net, alpha2 updates while (1-alpha2) remains
        :return:
        """
        if self.player == "white":
            self._soft_update(self.model_target, self.model_eval, self.alpha2)
            print('\nUpdate target_model weights')

    # @profile(stream=fp)
    def learn(self):
        """
        Trains the DQN model (white player only).

        DQN update rule:
            if terminal:  target_Q(s,a) = r
            else:         target_Q(s,a) = r + gamma * max_a' Q_target(s', a')

        Key design notes:
        - model_eval (online network) predicts Q(s) for current states — these are the
          values being optimised via gradient descent.
        - model_target (target network) predicts Q(s') for next states — these provide
          stable bootstrap targets and are updated periodically via soft copy.
        - Loss is computed only on the taken action via torch.gather, so gradients flow
          through the network correctly without spurious signal from unvisited actions.
        - When using PER, importance-sampling weights scale per-sample loss to correct
          for non-uniform sampling bias, and TD errors update priorities afterward.

        :return: loss value (float), or None if not enough transitions to learn
        """
        if self.player != "white":
            return

        if len(self.replay_buffer) < self.batch_size:
            return

        # sync model_eval and model_targets periodically (skip step 0 — no gradient update yet)
        if self.learn_step_counter > 0 and self.learn_step_counter % self.replace_target_iter == 0:
            self._tgt_evl_sync()

        # sample from replay buffer (uniform or prioritized)
        samples, is_weights, indices = self.replay_buffer.sample(self.batch_size)

        zipped_samples = list(zip(*samples))
        states, actions, rewards, new_states, dones = zipped_samples

        states_arr = np.array(states)
        new_states_arr = np.array(new_states)

        # convert to tensors on the target device
        states_t = torch.FloatTensor(states_arr).to(self.device)           # (batch, 1, 64)
        next_t = torch.FloatTensor(new_states_arr).to(self.device)         # (batch, 1, 64)
        actions_t = torch.LongTensor(list(actions)).to(self.device)        # (batch,)
        rewards_t = torch.FloatTensor(list(rewards)).to(self.device)       # (batch,)
        dones_t = torch.FloatTensor([float(d) for d in dones]).to(self.device)  # (batch,)

        # eval net predicts Q(s) for current states; gather selects only the taken action
        q_eval = self.model_eval(states_t)                                          # (batch, 64)
        q_taken = q_eval.gather(1, actions_t.unsqueeze(1)).squeeze(1)              # (batch,)

        # target net predicts Q(s') for next states (no gradients)
        with torch.no_grad():
            q_next = self.model_target(next_t)                                      # (batch, 64)
            q_next_max = q_next.max(dim=1)[0]                                       # (batch,)

        # Bellman TD target for the taken action
        td_targets = rewards_t + (1.0 - dones_t) * self.gamma * q_next_max        # (batch,)

        # TD errors for priority update in PER (computed before the gradient step)
        td_errors = (td_targets - q_taken).abs().detach().cpu().numpy()
        if indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors)

        # element-wise squared error, IS-weighted when using PER
        element_loss = (q_taken - td_targets) ** 2                                 # (batch,)
        if is_weights is not None:
            is_w = torch.FloatTensor(is_weights).to(self.device)
            loss = (is_w * element_loss).mean()
        else:
            loss = element_loss.mean()

        # single gradient step with gradient clipping (mirrors TF clipnorm=0.5)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model_eval.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()

        loss_val = loss.item()

        print("\nEpsilon:", round(self.epsilon, 4),
              "Replay Buffer:", len(self.replay_buffer),
              "Learn Step:", self.learn_step_counter,
              "Loss:", '%.4f' % loss_val,
              "\n")

        # increment the learning step counter
        self.learn_step_counter += 1

        # update the epsilon for epsilon greedy exploration / exploitation
        self.epsilon = max(cfg.agent_setting.EPSILON_MIN, self.epsilon * self.epsilon_reduce)

        return loss_val

    # @profile(stream=fp)
    def reward_transition_update(self, reward: float):
        """
        Patch white's last replay-buffer transition when black makes the game-ending move.

        When black plays last, white's most recent stored transition has an interim reward
        (possibly including a shaped component) and done=False. This method adds the
        terminal reward on top of the existing reward and marks done=True so the Q-update
        correctly uses ``target = reward`` instead of ``target = reward + gamma * max_Q(s')``.

        :param reward: terminal reward determined after the game ends
        :return: None
        """
        if self.player == "white":
            self.replay_buffer.update_last(reward, done=True)

    def save_model(self, name="OthelloDQN", save_step='training'):
        """
        Saves the target network weights as a PyTorch state dict (.pt).

        model_target is saved (not model_eval) because:
        - choose_action uses model_target for inference, so win-rate measurements reflect
          model_target's performance — saving model_eval would store a different, noisier network.
        - model_target is the temporally-smoothed snapshot, making it the correct artefact
          to checkpoint and deploy.
        On reload, load_model sets both model_eval and model_target to the saved weights,
        which is the correct starting state for both play and resumed training.
        """
        save_dir = "./models/{0}".format(save_step)
        os.makedirs(save_dir, exist_ok=True)
        pt_path = "{0}/{1}.pt".format(save_dir, name)
        torch.save(self.model_target.state_dict(), pt_path)
        self.model_full_path = pt_path

    def load_model(self, path="", name="OthelloDQN", format_type="model"):
        """
        loads weights and model from a PyTorch .pt state dict file
        :return:
        """
        if not os.path.exists(path):
            sys.exit("cannot load %s" % name)

        try:
            if path.endswith(".pt") and os.path.isfile(path):
                pt_path = path
            else:
                pt_path = "{0}/{1}.pt".format(path, name)
            print(pt_path)
            state_dict = torch.load(pt_path, map_location=self.device, weights_only=True)
            if hasattr(self, "model_eval"):
                self.model_eval.load_state_dict(state_dict)
            self.model_target.load_state_dict(state_dict)
            self.model_target.eval()
            self.model_full_path = pt_path
            return True, "Successfully loaded agent from\n{0}".format(self.model_full_path)
        except (ValueError, RuntimeError) as ve:
            error_str = str(ve)
            print(error_str)
            return False, "Failed to load agent!"

    def reload_model(self, path=None):
        """
        reloads previously selected weights and model
        :return:
        """
        # if no new model specified then use previously loaded model
        if path is None:
            path = self.model_full_path

        # check that the model exists
        if not os.path.exists(path):
            sys.exit("cannot load %s" % path)

        # load model
        try:
            state_dict = torch.load(self.model_full_path, map_location=self.device, weights_only=True)
            self.model_eval.load_state_dict(state_dict)
            self.model_target.load_state_dict(state_dict)
            self.model_target.eval()
            return True, "Successfully loaded agent from\n{0}".format(self.model_full_path)
        except (ValueError, OSError, RuntimeError) as ve:
            error_str = str(ve)
            print(error_str)
            return False, "Failed to load agent!"
