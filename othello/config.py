# centralized configuration file.
# It can be updated using argparse as well.


class agent_setting:
    # ACTION_DIM = 64
    # STATE_DIM = 64

    GAMMA = 0.9975  # reward decay rate - 0.975, 0.9975
    ALPHA1 = 0.3  # soft copy weights for self-play, alpha1 updates while (1-alpha1) remains
    ALPHA2 = 0.1  # soft copy weights from eval net to target net, alpha2 updates from eval while (1-alpha2) remains for target net
    EPSILON_REDUCE = 0.999975  # 0.995, 0.9995, 0.99975, 0.9999, 0.999975
    EPSILON = 1.0  # epsilon parameter for epsilon greedy selection
    EPSILON_MIN = 0.075  # minimum exploration rate — prevents full exploitation collapse

    # q network learning parameters
    LEARNING_RATE = 0.0001  # 0.001, 0.0005, 0.0001, 0.00005
    BATCH_SIZE = 5120  # 128, 256, 512, 768, 1024, 2048, 4096, 5120, 10240

    # total learning step - count how many times the eval net has been updated, used to set a basis for updating
    # the target net
    LEARN_STEP_COUNTER = 0
    REPLACE_TARGET_ITER = 200  # 150, 200, 250, 300

    # replay buffer settings
    REPLAY_BUFFER_SIZE = 200000  # 20000, 40000, 75000, 150000
    REPLAY_BUFFER_STRATEGY = "uniform"  # "uniform" or "per"

    # PER hyperparameters (only used when REPLAY_BUFFER_STRATEGY = "per")
    PER_ALPHA = 0.6  # prioritization exponent [0=uniform, 1=full priority]
    PER_BETA = 0.4  # IS weight exponent (anneals to 1.0)
    PER_BETA_INCREMENT = 0.001
    PER_EPSILON = 0.01  # small constant to avoid zero priority
    PER_ABS_ERR_UPPER = 10.0  # TD error clip (accommodates reward range)

    # penalty and reward
    PENALTY = -15
    REWARD = 25
    TIE = 5

    # reward shaping (potential-based)
    REWARD_SHAPING_ENABLED = True
    REWARD_POSITIONAL_WEIGHT = 0.01  # positional value scaling


class training_param:
    EPOCHS = 150000
    EPOCH_WIN_RATE_LOG = 50
    SELF_PLAY_UPDATE_LOG = 5000
    WARMUP_EPOCHS = 15000  # epochs vs random before switching to self-play (curriculum mode only)
    SELF_PLAY_UPDATE_WIN_THRESHOLD = 0.50  # minimum win rate required before opponent weights are updated
