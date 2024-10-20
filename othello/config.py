# centralized configuration file.
# It can be updated using argparse as well.


class agent_setting:
    # ACTION_DIM = 64
    # STATE_DIM = 64

    GAMMA = 0.95  # reward decay rate - 0.975, 0.9975
    ALPHA1 = 0.3  # soft copy weights for self-play, alpha1 updates while (1-alpha1) remains
    ALPHA2 = 0.45  # soft copy weights from eval net to target net, alpha2 updates from eval while (1-alpha2) remains for target net
    EPSILON_REDUCE = 0.9999  # 0.995, 0.9995, 0.99975, 0.9999, 0.999975
    EPSILON = 1.0  # epsilon parameter for epsilon greedy selection

    # q network learning parameters
    LEARNING_RATE = 0.0001  # 0.001, 0.0005, 0.0001, 0.00005
    BATCH_SIZE = 10240  # 128, 256, 512, 768, 1024, 2048, 4096, 10240
    TRAINING_EPOCHS = 50  # 15, 20, 50, 100

    # total learning step - count how many times the eval net has been updated, used to set a basis for updating
    # the target net
    LEARN_STEP_COUNTER = 0
    REPLACE_TARGET_ITER = 75  # 10, 50, 75, 100, 150

    # replay buffer settings
    REPLAY_BUFFER_SIZE = 200000  # 20000, 40000, 75000, 150000

    # penalty and reward
    PENALTY = -15
    REWARD = 25
    TIE = 5


class training_param:
    EPOCHS = 150000
    EPOCH_WIN_RATE_LOG = 50
    SELF_PLAY_UPDATE_LOG = 5000
