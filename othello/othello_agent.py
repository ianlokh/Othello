import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Class for the deep neural network model — PyTorch port of the TensorFlow Sequential model.

    Architecture exactly mirrors the original TF build_model():
        Dense(64, relu) → Dense(64, relu)
        Dense(64) → LayerNorm → LeakyReLU
        Dense(128, relu) × 3
        Dense(64) → LayerNorm → LeakyReLU
        Dense(64, relu) → Dense(action_dim, linear)

    Optimizer: Adam with ExponentialDecay LR schedule (staircase=True, decay_steps=10000,
               decay_rate=0.99) and gradient clipping (norm ≤ 0.5).
    Loss:      MSELoss — equivalent to tf.keras.losses.MeanSquaredError().
    """

    def __init__(self, nb_observations, action_dim, learning_rate=None):
        super().__init__()
        self.nb_observations = nb_observations
        self.action_dim = action_dim
        # Fall back to config value when called without explicit learning_rate
        self.learning_rate = learning_rate if learning_rate is not None else cfg.agent_setting.LEARNING_RATE

        # Build network in __init__ so the model is immediately usable for inference
        # even before build_model() is called (e.g. for shape checks or weight loading).
        self.net = nn.Sequential(
            nn.Linear(nb_observations, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),

            # Dense(128, relu) × 3
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),

            nn.Linear(64, 64), nn.ReLU(),
            # Output: linear activation (no softmax — applied post-hoc in choose_action)
            nn.Linear(64, action_dim),
        )

        # Optimizer / scheduler / criterion are wired up in build_model()
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def residual_block(self, inputs, filters):
        """
        Define a residual block with two convolutional layers.
        (Currently unused — retained for API compatibility.)

        :param inputs: Input tensor to the block.
        :param filters: Number of filters for the convolutions.
        :return: nn.Sequential block (skip connection must be applied externally).
        """
        return nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
        )

    def build_model(self):
        """
        Wire up the optimizer, LR scheduler, and loss function, then return self.

        Usage mirrors TF:  model = OthelloDQNModel(...).build_model()

        Optimizer: Adam with ExponentialDecay (staircase=True) — identical to TF:
            lr = initial_lr * 0.99 ^ floor(step / 10_000)
            PyTorch equivalent: StepLR(step_size=10_000, gamma=0.99),
            called once per train_on_batch invocation.
        Gradient clipping: norm ≤ 0.5 (applied inside train_on_batch, mirrors TF clipnorm=0.5).
        Loss: MSELoss (matches tf.keras.losses.MeanSquaredError).

        :return: self
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # StepLR decays by gamma every step_size scheduler.step() calls,
        # which equals one train_on_batch call → matches TF ExponentialDecay staircase.
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.99
        )
        self.criterion = nn.MSELoss()
        return self

    def forward(self, x):
        """
        Forward pass. Handles both 2-D and 3-D inputs to remain compatible with the
        TF model's input_shape=(1, nb_observations) convention used in OthelloDQN.

        Input shapes accepted:
            (batch, nb_observations)      → output (batch, action_dim)
            (batch, 1, nb_observations)   → output (batch, 1, action_dim)

        The 3-D path preserves the middle dimension so that OthelloDQN.learn()'s
        indexing pattern  targets[i][0][action]  continues to work unchanged.

        :param x: numpy array or torch.Tensor
        :return: torch.Tensor of Q-values
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        is_3d = (x.dim() == 3)
        if is_3d:
            x = x.squeeze(1)       # (batch, 1, obs) → (batch, obs)

        out = self.net(x)          # (batch, action_dim)

        if is_3d:
            out = out.unsqueeze(1)  # (batch, action_dim) → (batch, 1, action_dim)

        return out

    def predict_on_batch(self, x):
        """
        Run a forward pass in eval / no-grad mode and return a numpy array.
        Output shape mirrors forward(): preserves 3-D structure if input is 3-D.

        :param x: numpy array or torch.Tensor
        :return: numpy array of Q-values
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        return out.cpu().numpy()

    def train_on_batch(self, x, y, sample_weight=None):
        """
        Perform one gradient step.

        Matches TF model_eval.train_on_batch(states, targets, sample_weight=is_weights):
        - Computes MSE loss, optionally weighted per-sample for PER.
        - Clips gradient norm to 0.5 (TF clipnorm=0.5).
        - Steps both optimizer and LR scheduler.

        :param x: numpy array of states, shape (batch, 1, obs) or (batch, obs)
        :param y: numpy array of Q-value targets, same shape as forward output
        :param sample_weight: optional numpy array of IS weights, shape (batch,)
        :return: (loss_value: float, metric: float)  — metric is 0.0 (accuracy not applicable)
        """
        self.train()
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        self.optimizer.zero_grad()
        pred = self(x_t)

        # Per-element MSE, then reduce with optional IS weighting
        loss_per_elem = F.mse_loss(pred, y_t, reduction='none')

        if sample_weight is not None:
            w = torch.tensor(sample_weight, dtype=torch.float32)
            # Average over all non-batch dims, then take weighted mean over batch
            loss = (w * loss_per_elem.flatten(start_dim=1).mean(dim=1)).mean()
        else:
            loss = loss_per_elem.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), 0.0

    def get_weights(self):
        """
        Return model parameters as a list of numpy arrays.
        Used by OthelloDQN to sync eval → target network weights.

        :return: list of numpy arrays, one per parameter tensor
        """
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights):
        """
        Load parameters from a list of numpy arrays (produced by get_weights()).

        :param weights: list of numpy arrays matching the parameter order
        """
        with torch.no_grad():
            for param, w in zip(self.parameters(), weights):
                param.data.copy_(torch.tensor(w, dtype=torch.float32))

    def save(self, path):
        """
        Save the model state dict to *path*.

        :param path: file path (conventionally ending in .pt)
        """
        torch.save(self.state_dict(), path)

    def save_weights(self, path, overwrite=True):
        """
        Save the model state dict to *path* (alias for save()).

        :param path: file path
        :param overwrite: ignored (always overwrites, matching TF behaviour)
        """
        torch.save(self.state_dict(), path)


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

        # only white player learns hence q network will only be created for white player
        if self.player == "white":
            # self.model_eval = self.build_model(nb_observations)  # this is the q network
            self.model_eval = OthelloDQNModel(nb_observations, self.action_dim,
                                              self.learning_rate).build_model()  # this is the q network
            self.model_eval.train()   # online network stays in training mode

        # regardless of training (random or self-play) target network will always be created because this is the network
        # that will be used to predict the action
        self.model_target = OthelloDQNModel(nb_observations, self.action_dim,
                                            self.learning_rate).build_model()  # this is the target network
        self.model_target.eval()      # target network stays in inference mode

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
        np.random.seed(seed)
        print("Seed:{:d}".format(seed))

    def set_global_determinism(self, seed):
        """
        sets tensorflow specific deterministic parameters for reproducibility
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
            observation = np.expand_dims(observation, axis=0)  # (1, 64, )

            prediction = self.model_target.predict_on_batch(observation)

            prediction = softmax(np.ma.array(prediction, mask=mask).filled(fill_value=-1e9), axis=None)

            # action = tf.argmax(prediction[0], axis=1)
            # action = int(tf.keras.backend.eval(action))
            action = np.argmax(prediction[0], axis=1).item()

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
        with torch.no_grad():
            for t_param, s_param in zip(target_model.parameters(), source_model.parameters()):
                t_param.data.mul_(1 - alpha).add_(s_param.data * alpha)

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
        - A single gradient step is taken per call (standard DQN). Multiple steps on
          the same precomputed targets would overfit on stale values.
        - When using PER, importance-sampling weights scale per-sample loss to correct
          for non-uniform sampling bias, and TD errors update priorities afterward.

        :return: None
        """
        if self.player != "white":
            return

        if len(self.replay_buffer) < self.batch_size:
            return

        # sync model_eval and model_targets periodically
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._tgt_evl_sync()

        # sample from replay buffer (uniform or prioritized)
        samples, is_weights, indices = self.replay_buffer.sample(self.batch_size)

        zipped_samples = list(zip(*samples))
        states, actions, rewards, new_states, dones = zipped_samples

        states_arr = np.array(states)
        new_states_arr = np.array(new_states)

        # eval net predicts Q(s) for current states (these are the values being trained)
        # target net predicts Q(s') for next states (stable bootstrap targets)
        targets = np.array(self.model_eval.predict_on_batch(states_arr))
        q_values_next = np.array(self.model_target.predict_on_batch(new_states_arr))

        # build target batch and compute TD errors for priority updates
        td_errors = np.zeros(self.batch_size, dtype=np.float32)
        target_batch = []

        for i in range(self.batch_size):
            q_next_max = np.max(q_values_next[i][0])
            target = targets[i].copy()
            old_q = target[0][actions[i]]

            if dones[i]:
                target[0][actions[i]] = rewards[i]
            else:
                target[0][actions[i]] = rewards[i] + q_next_max * self.gamma

            td_errors[i] = abs(target[0][actions[i]] - old_q)
            target_batch.append(target)

        # update priorities in PER buffer (no-op for uniform)
        if indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors)

        # single gradient step (IS-weighted when using PER)
        targets_arr = np.array(target_batch)
        if indices is not None:
            loss, metrics = self.model_eval.train_on_batch(
                states_arr, targets_arr, sample_weight=is_weights
            )
        else:
            loss, metrics = self.model_eval.train_on_batch(states_arr, targets_arr)

        print("\nEpsilon:", round(self.epsilon, 4),
              "Replay Buffer:", len(self.replay_buffer),
              "Learn Step:", self.learn_step_counter,
              "Loss:", '%.4f' % loss,
              "Metrics:", '%.4f' % metrics,
              "\n")

        # increment the learning step counter
        self.learn_step_counter += 1

        # update the epsilon for epsilon greedy exploration / exploitation
        self.epsilon = max(cfg.agent_setting.EPSILON_MIN, self.epsilon * self.epsilon_reduce)

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
        Saves the target network weights and model.

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
        model_path = "{0}/{1}.pt".format(save_dir, name)
        self.model_target.save(model_path)
        self.model_full_path = model_path   # enables reload_model() to find the saved file

    def load_model(self, path="", name="OthelloDQN", format_type="model"):
        """
        loads weights and model
        :return:
        """
        if not os.path.exists(path):
            sys.exit("cannot load %s" % name)

        try:
            # Resolve model path — accept a direct .pt file or a directory + name
            if path.endswith(".pt") and os.path.isfile(path):
                model_path = path
            else:
                model_path = "{0}/{1}.pt".format(path, name)
            print(model_path)
            state_dict = torch.load(model_path, weights_only=True)
            self.model_eval.load_state_dict(state_dict)
            self.model_full_path = model_path

            self.model_target.set_weights(self.model_eval.get_weights())
            self.model_target.eval()
            return True, "Successfully loaded agent from\n{0}".format(self.model_full_path)
        except (ValueError, RuntimeError, FileNotFoundError) as ve:
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
            state_dict = torch.load(self.model_full_path, weights_only=True)
            self.model_eval.load_state_dict(state_dict)
            self.model_target.set_weights(self.model_eval.get_weights())
            self.model_target.eval()
            return True, "Successfully loaded agent from\n{0}".format(self.model_full_path)
        except (ValueError, RuntimeError, OSError) as ve:
            error_str = str(ve)
            print(error_str)
            return False, "Failed to load agent!"
