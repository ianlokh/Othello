# centralized configuration file.
# It can be updated using argparse as well.


class agent_setting:
    ACTION_DIM = 64
    STATE_DIM = 64

    GAMMA = 0.95  # reward decay rate
    ALPHA1 = 0.1  # soft copy weights from white to black, alpha1 updates while (1-alpha1) remains
    ALPHA2 = 0.45  # soft copy weights from eval net to target net, alpha2 updates while (1-alpha2) remains
    EPSILON_REDUCE = 0.99975  # 0.995, 0.9995, 0.99975
    EPSILON = 1.0

    # q network learning parameters
    LEARNING_RATE = 0.0005  # 0.001, 0.0005, 0.0001
    BATCH_SIZE = 128  # 128, 256, 512, 768, 1024, 2048
    TRAINING_EPOCHS = 50  # 15, 20, 50, 100

    # total learning step - count how many times the eval net has been updated, used to set a basis for updating
    # the target net
    LEARN_STEP_COUNTER = 0
    REPLACE_TARGET_ITER = 75  # 10, 50, 75, 100, 150

    # replay buffer settings
    REPLAY_BUFFER_SIZE = 75000  # 20000, 40000, 75000
