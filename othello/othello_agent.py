import os
import random
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy.special import softmax

from othello import config as cfg
from othello.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer

# for performance profiling
# import cProfile as cprofile
# from memory_profiler import profile
# fp = open("report-agent.log", "w+")  # to capture memory profile logs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

class OthelloDQNModel:
    """
    Class for the deep neural network model
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
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)  # No activation here
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()([x, inputs])  # Skip connection

        return tf.keras.layers.Activation('relu')(x)

    def build_model(self):
        """
        build tensorflow model
        :return: tensorflow model
        """
        def root_mean_squared_log_error(y_true, y_pred):
            msle = tf.keras.losses.MeanSquaredLogarithmicError()
            return K.sqrt(msle(y_true, y_pred))

        def root_mean_squared_error(y_true, y_pred):
            mse = tf.keras.losses.MeanSquaredError()
            return K.sqrt(mse(y_true, y_pred))

        _model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(1, self.nb_observations), activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),

            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            # tf.keras.layers.Dense(128, activation="relu"),
            # tf.keras.layers.Dropout(rate=0.3),
            # tf.keras.layers.Dense(128, activation="relu"),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),

            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Dense(64, activation="relu"),
            # don't need softmax here because the subsequent post-processing will do the softmax
            # tf.keras.layers.Dense(self.action_dim, activation=tf.keras.activations.softmax)
            tf.keras.layers.Dense(self.action_dim, activation=tf.keras.activations.linear)

            # Notes:
            # If you are training a binary classifier you can solve the problem with sigmoid activation + binary crossentropy loss.
            # If you are training a multi-class classifier with multiple classes, then you need softmax activation + crossentropy loss.
            # If you are training a regressor you need a proper activation function with MSE or MAE loss,
            # usually.With "proper" I mean linear, in case your output is unbounded, or ReLU in case your output
            # takes only positive values.

        ])

        # The following metrics and losses do not work
        # tf.keras.metrics.sparse_categorical_accuracy
        # tf.keras.metrics.sparse_categorical_crossentropy
        # tf.keras.losses.MeanAbsolutePercentageError()

        # Decay LR by 10% every 10,000 learn steps (one step = one train_on_batch call).
        # Over 75,000 epochs this brings LR from 1e-4 down to ~4.8e-5, helping the
        # network fine-tune later in training without overwriting earlier Q-values.
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=True,
        )

        _model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                                          clipnorm=0.5),
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['accuracy'])
        return _model


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

        # regardless of training (random or self-play) target network will always be created because this is the network
        # that will be used to predict the action
        self.model_target = OthelloDQNModel(nb_observations, self.action_dim,
                                            self.learning_rate).build_model()  # this is the target network

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
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print("Seed:{:d}".format(seed))

    def set_global_determinism(self, seed):
        """
        sets tensorflow specific deterministic parameters for reproducibility
        :param seed:
        :return:
        """
        self.set_env_seeds(seed=seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

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

            with tf.device('/cpu:0'):
                prediction = self.model_target(observation, training=False).numpy()

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
        for t_var, s_var in zip(target_model.variables, source_model.variables):
            t_var.assign(t_var * (1 - alpha) + s_var * alpha)

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
        Trains the DDQN model (white player only).

        DDQN update rule:
            if terminal:  target_Q(s,a) = r
            else:         target_Q(s,a) = r + gamma * Q_target(s', argmax_a' Q_eval(s', a'))

        Key design notes:
        - model_eval (online network) predicts Q(s) for current states — these are the
          values being optimised via gradient descent. It also selects the best next
          action (argmax) to decouple action selection from evaluation (DDQN).
        - model_target (target network) evaluates Q(s') at the action chosen by
          model_eval — this provides stable bootstrap targets and reduces overestimation
          bias compared to plain DQN. Updated periodically via soft copy.
        - A single gradient step is taken per call. Multiple steps on
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
        # eval net also predicts Q(s') for action selection only (DDQN: decouple select from evaluate)
        targets = np.array(self.model_eval.predict_on_batch(states_arr))
        q_values_next = np.array(self.model_target.predict_on_batch(new_states_arr))
        q_values_next_eval = np.array(self.model_eval.predict_on_batch(new_states_arr))

        # build target batch and compute TD errors for priority updates
        td_errors = np.zeros(self.batch_size, dtype=np.float32)
        target_batch = []

        for i in range(self.batch_size):
            best_next_action = np.argmax(q_values_next_eval[i][0])  # eval net selects action
            q_next_max = q_values_next[i][0][best_next_action]       # target net evaluates it
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
        self.model_target.save_weights("{0}/{1}.weights.h5".format(save_dir, name), overwrite=True)
        self.model_target.save("{0}/{1}_model.keras".format(save_dir, name))

    def load_model(self, path="", name="OthelloDQN", format_type="model"):
        """
        loads weights and model
        :return:
        """
        if not os.path.exists(path):
            sys.exit("cannot load %s" % name)

        try:
            if format_type == "model":
                # If the user selected the .keras file directly, use it as-is;
                # otherwise treat path as a directory and append the filename.
                if path.endswith(".keras") and os.path.isfile(path):
                    model_path = path
                else:
                    model_path = "{0}/{1}_model.keras".format(path, name)
                print(model_path)
                self.model_eval = tf.keras.models.load_model(model_path)
                self.model_full_path = model_path
            elif format_type == "weights":
                if path.endswith(".weights.h5") and os.path.isfile(path):
                    weights_path = path
                else:
                    weights_path = "{0}/{1}.weights.h5".format(path, name)
                print(weights_path)
                self.model_eval.load_weights(weights_path)
                self.model_full_path = weights_path

            self.model_target.set_weights(self.model_eval.get_weights())
            return True, "Successfully loaded agent from\n{0}".format(self.model_full_path)
        except ValueError as ve:
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
            self.model_eval = tf.keras.models.load_model(self.model_full_path)
            self.model_target.set_weights(self.model_eval.get_weights())
            return True, "Successfully loaded agent from\n{0}".format(self.model_full_path)
        except (ValueError, OSError) as ve:
            error_str = str(ve)
            print(error_str)
            return False, "Failed to load agent!"
