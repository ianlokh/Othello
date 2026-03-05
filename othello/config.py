# Centralised configuration for the Othello DQN agent and training loop.
# Parameters are grouped by concern. Values that have been empirically tested
# are shown in the inline comments for reference.


class agent_setting:
    """Hyperparameters for the OthelloDQN agent.

    Parameter groups
    ----------------
    1. Bellman / Q-learning core      — GAMMA
    2. Target network update          — ALPHA2, REPLACE_TARGET_ITER, LEARN_STEP_COUNTER
    3. Self-play opponent update      — ALPHA1
    4. Exploration (ε-greedy)         — EPSILON, EPSILON_MIN, EPSILON_REDUCE
    5. Optimizer / network training   — LEARNING_RATE, BATCH_SIZE
    6. Replay buffer — core           — REPLAY_BUFFER_SIZE, REPLAY_BUFFER_STRATEGY
    7. Replay buffer — PER            — PER_* (only active when strategy = "per")
    8. Terminal rewards               — REWARD, PENALTY, TIE
    9. Reward shaping                 — REWARD_SHAPING_ENABLED, REWARD_POSITIONAL_WEIGHT
    """

    # -------------------------------------------------------------------------
    # 1. Bellman / Q-learning core
    # -------------------------------------------------------------------------
    # GAMMA is the discount factor applied to future Q-values in the Bellman
    # equation:  Q(s,a) = r + GAMMA * max Q(s', a')
    #
    # Othello rationale: rewards are sparse and terminal (win/loss only at game
    # end). With up to 60 moves per game (white plays ~30), GAMMA must be high
    # enough to propagate the terminal signal back through the entire trajectory.
    #
    #   GAMMA = 0.9975 → terminal reward retains ~86% of value over 60 moves.
    #   GAMMA = 0.97   → terminal reward retains only ~16% over 60 moves (too low).
    #
    # Tried: 0.975, 0.9975

    GAMMA = 0.9975

    # -------------------------------------------------------------------------
    # 2. Target network update
    # -------------------------------------------------------------------------
    # The target network (model_target) provides stable Bellman bootstrap
    # targets during learning and is also the network that plays the game.
    # It is periodically soft-updated from the eval network (model_eval):
    #
    #   model_target = (1 - ALPHA2) * model_target + ALPHA2 * model_eval
    #
    # Two parameters jointly control the update:
    #   ALPHA2              — fraction copied per sync (jump size).
    #   REPLACE_TARGET_ITER — how many learn steps between each sync (frequency).
    #
    # The effective per-step tracking rate is:
    #   effective_tau = ALPHA2 / REPLACE_TARGET_ITER
    #
    # Design rules:
    #   • Keep ALPHA2 ≤ 0.05  →  each sync shifts target by at most 5%, preventing
    #     large sudden jumps that destabilise Bellman targets for the next ITER steps.
    #   • effective_tau ≈ 0.0003–0.001  →  conservative tracking appropriate for
    #     sparse-reward board games.
    #   • Current: effective_tau = 0.05 / 100 = 0.0005 per step.
    #
    # Tried REPLACE_TARGET_ITER: 150, 200, 250, 300 (now 100)

    ALPHA2 = 0.05               # jump size per sync; keep ≤ 0.05
    REPLACE_TARGET_ITER = 100   # steps between target syncs
    LEARN_STEP_COUNTER = 0      # initial value — incremented inside learn(); do not change

    # -------------------------------------------------------------------------
    # 3. Self-play opponent update
    # -------------------------------------------------------------------------
    # ALPHA1 is the soft-copy fraction used by assign_weights() to update the
    # self-play opponent (agent_other) from the trained agent (agent_white):
    #
    #   agent_other.model_target = (1 - ALPHA1) * agent_other + ALPHA1 * agent_white.model_eval
    #
    # NOTE: ALPHA1 is currently UNUSED. The training loop (othello_train.py)
    # performs a hard copy directly and does not call assign_weights(). Retained
    # for future use if a soft opponent update is preferred over a hard copy.
    #
    # If activated: 0.3 blends 30% of the trained agent into the opponent per
    # update, producing a gradual curriculum ramp rather than a hard reset.

    ALPHA1 = 0.3                # currently unused — see note above

    # -------------------------------------------------------------------------
    # 4. Exploration — ε-greedy
    # -------------------------------------------------------------------------
    # The agent selects a random valid action with probability ε (exploration)
    # and the greedy Q-value action with probability 1-ε (exploitation).
    #
    # ε decays multiplicatively after every learn() call:
    #   ε_new = max(EPSILON_MIN, ε * EPSILON_REDUCE)
    #
    # Decay trajectory (EPSILON_REDUCE = 0.999975 per learn step):
    #   Step  75,000 → ε ≈ 0.153   (EPSILON_MIN not reached — pair with 75k epochs)
    #   Step  94,155 → ε = EPSILON_MIN = 0.095
    #   Step 150,000 → ε = EPSILON_MIN = 0.095 (floor active from step ~94k onward)
    #
    # IMPORTANT: pair EPSILON_REDUCE with EPOCHS so EPSILON_MIN is reached
    # during training. For 75k epochs use a faster decay (e.g. 0.99995).
    # For 150k epochs the current value is correctly calibrated.
    #
    # Tried EPSILON_REDUCE: 0.995, 0.9995, 0.99975, 0.9999, 0.999975

    EPSILON = 1.0               # initial exploration rate (100% random at start)
    EPSILON_MIN = 0.095         # floor — maintains 9.5% exploration throughout
    EPSILON_REDUCE = 0.999975   # per-step decay multiplier; calibrated for 150k epochs

    # -------------------------------------------------------------------------
    # 5. Optimizer / network training
    # -------------------------------------------------------------------------
    # LEARNING_RATE: Adam optimiser initial LR. An exponential decay schedule
    # reduces it by 10% every 10,000 learn steps (hardcoded in OthelloDQNModel),
    # bringing it from 1e-4 to ~4.8e-5 over 75k steps.
    #
    # BATCH_SIZE: number of transitions sampled per learn() call. A large batch
    # (2048) is chosen because the Pygame environment is slow — fewer, larger
    # gradient steps are more efficient than many small ones.
    #
    # Constraint: REPLAY_BUFFER_SIZE must be >> BATCH_SIZE.
    # Current ratio: 400,000 / 2,048 ≈ 195× (well above the practical minimum of ~10×).
    # Learning starts once the buffer holds ≥ BATCH_SIZE transitions.
    #
    # Tried LEARNING_RATE: 0.001, 0.0005, 0.0001, 0.00005
    # Tried BATCH_SIZE: 128, 256, 512, 768, 1024, 2048, 4096, 5120, 10240

    LEARNING_RATE = 0.0001      # Adam initial LR; decays inside OthelloDQNModel
    BATCH_SIZE = 2048           # transitions per gradient step

    # -------------------------------------------------------------------------
    # 6. Replay buffer — core
    # -------------------------------------------------------------------------
    # The replay buffer stores past transitions (s, a, r, s', done) so the
    # agent can learn from decorrelated, off-policy experience.
    #
    # REPLAY_BUFFER_SIZE sizing rules:
    #   • White stores ~30 transitions per episode (half of 60-move game).
    #   • Buffer should flush warmup (random-opponent) transitions within ~10%
    #     of the self-play phase to avoid stale-policy contamination.
    #   • Pair with EPOCHS:
    #       75k  epochs → 200,000  (warmup transitions evicted by epoch ~16,667)
    #       150k epochs → 400,000  (warmup transitions evicted by epoch ~23,333)
    #
    # REPLAY_BUFFER_STRATEGY:
    #   "uniform" — every stored transition is equally likely to be sampled.
    #   "per"     — transitions with high TD error are sampled more often
    #               (Prioritized Experience Replay, Schaul et al. 2016).
    #               Requires correct PER_BETA_INCREMENT calibration — see section 7.
    #
    # Tried REPLAY_BUFFER_SIZE: 200,000  400,000  750,000  1,500,000

    REPLAY_BUFFER_SIZE = 400000         # sized for 150k-epoch run
    REPLAY_BUFFER_STRATEGY = "uniform"  # "uniform" or "per"

    # -------------------------------------------------------------------------
    # 7. Replay buffer — Prioritized Experience Replay (PER)
    # -------------------------------------------------------------------------
    # Only active when REPLAY_BUFFER_STRATEGY = "per".
    # Reference: Schaul et al. (2016) "Prioritized Experience Replay".
    #
    # PER_ALPHA:          prioritization exponent.
    #                     0 = uniform sampling (ignores TD error).
    #                     1 = fully proportional to TD error.
    #                     0.6 is the standard from the original paper.
    #
    # PER_BETA:           importance-sampling (IS) weight exponent.
    #                     Corrects for the bias introduced by non-uniform sampling.
    #                     Anneals from PER_BETA → 1.0 over training so that
    #                     IS correction is complete by the end of training.
    #
    # PER_BETA_INCREMENT: amount added to PER_BETA each time sample() is called.
    #                     CRITICAL — must be calibrated to EPOCHS:
    #                       correct value = (1.0 - PER_BETA) / EPOCHS
    #                       for 150k epochs: (1.0 - 0.4) / 150,000 = 0.000004
    #                     The current value of 0.001 reaches beta=1.0 after only
    #                     ~600 steps — far too fast. Fix before enabling PER.
    #
    # PER_EPSILON:        small constant added to every TD error to prevent any
    #                     transition from having zero priority and never being sampled.
    #
    # PER_ABS_ERR_UPPER:  clips very large TD errors to prevent a single
    #                     surprising transition from dominating the buffer.
    #                     Max possible TD error = REWARD - PENALTY = 25-(-15) = 40,
    #                     so 10.0 clips to the upper quartile of the error range.

    PER_ALPHA = 0.6             # prioritization exponent (0=uniform, 1=full)
    PER_BETA = 0.4              # IS weight exponent; anneals to 1.0 over training
    PER_BETA_INCREMENT = 0.000004  # calibrated for 150k epochs: 0.6/150,000
    PER_EPSILON = 0.01          # min priority floor (avoids zero-priority transitions)
    PER_ABS_ERR_UPPER = 10.0    # TD error clip; max theoretical error = 40

    # -------------------------------------------------------------------------
    # 8. Terminal rewards
    # -------------------------------------------------------------------------
    # Applied at game end. Only one outcome reward is given per episode.
    #
    # Asymmetric design (|REWARD| > |PENALTY|) is intentional:
    #   • Encourages aggressive, winning-focused play.
    #   • The agent is penalised for losing but rewarded more for winning,
    #     so it learns to prefer winning over merely avoiding loss.
    # TIE is positive: a draw is mildly preferred over a loss.
    #
    # These values directly set the scale of Q-values in the network.
    # Reward shaping (section 9) adds smaller intermediate signals on top.

    REWARD = 25                 # win
    PENALTY = -15               # loss
    TIE = 5                     # draw

    # -------------------------------------------------------------------------
    # 9. Reward shaping — Potential-Based Reward Shaping (PBRS)
    # -------------------------------------------------------------------------
    # Intermediate shaping signal added to each non-terminal transition to guide
    # the agent toward positionally strong moves before the terminal reward arrives.
    # Uses the theory from Ng et al. (1999) which guarantees that PBRS preserves
    # the optimal policy of the original (unshaped) reward function.
    #
    # Shaping formula applied in store_transition():
    #   F(s, s') = GAMMA * Phi(s') - Phi(s)
    #   shaped_reward = reward + REWARD_POSITIONAL_WEIGHT * F(s, s')
    #
    # where Phi(s) = dot(POSITION_WEIGHTS, board_state) uses the classic Othello
    # positional weight matrix (corners=120, X-squares=-40, C-squares=-20, etc.)
    #
    # REWARD_POSITIONAL_WEIGHT scales the shaping signal relative to the terminal
    # rewards. At 0.01 the maximum shaped component is roughly ±5, which is
    # within the same order of magnitude as TIE (5) but well below REWARD (25).
    # Setting to 0.0 disables shaping without changing the flag.

    REWARD_SHAPING_ENABLED = True
    REWARD_POSITIONAL_WEIGHT = 0.01     # shaping scale; max contribution ≈ ±5


class training_param:
    """Hyperparameters for the outer training loop (othello_train.py).

    Parameter groups
    ----------------
    1. Training duration   — EPOCHS
    2. Logging             — EPOCH_WIN_RATE_LOG
    3. Curriculum / self-play — WARMUP_EPOCHS, SELF_PLAY_UPDATE_LOG,
                                SELF_PLAY_UPDATE_WIN_THRESHOLD
    """

    # -------------------------------------------------------------------------
    # 1. Training duration
    # -------------------------------------------------------------------------
    # Total number of episodes (games) to train for.
    # Pair with agent_setting.REPLAY_BUFFER_SIZE and EPSILON_REDUCE:
    #   75k  epochs → REPLAY_BUFFER_SIZE=200,000  EPSILON_REDUCE≈0.99995
    #   150k epochs → REPLAY_BUFFER_SIZE=400,000  EPSILON_REDUCE=0.999975

    EPOCHS = 150000             # 75000, 150000

    # -------------------------------------------------------------------------
    # 2. Logging
    # -------------------------------------------------------------------------
    # Win rate is computed over the last EPOCH_WIN_RATE_LOG episodes and logged
    # to the screen and Pygame display. The model is checkpointed whenever a new
    # best win rate is achieved at a logging boundary.

    EPOCH_WIN_RATE_LOG = 50     # episodes per win-rate window

    # -------------------------------------------------------------------------
    # 3. Curriculum / self-play
    # -------------------------------------------------------------------------
    # Training uses a two-phase curriculum:
    #   Phase 1 (warmup): agent_white plays against a random opponent for
    #     WARMUP_EPOCHS episodes to learn basic Othello strategy.
    #   Phase 2 (self-play): agent_white plays against a frozen copy of itself
    #     (agent_other) that is periodically updated when agent_white improves.
    #
    # WARMUP_EPOCHS: number of random-opponent episodes before switching.
    #   At ~30 transitions/episode this generates ~300k warmup transitions,
    #   which fills a 200k buffer or half-fills a 400k buffer.
    #
    # SELF_PLAY_UPDATE_LOG: episodes between opponent weight-copy checks.
    #   The copy only fires if the recent win rate meets the threshold below.
    #   With EPOCHS=150k and warmup=10k there are ~28 potential update windows.
    #
    # SELF_PLAY_UPDATE_WIN_THRESHOLD: minimum win rate required to promote
    #   agent_white's weights into agent_other. Prevents copying a regressing
    #   policy into the opponent. Set to 0.50 — agent must win at least half
    #   its recent games before the opponent is made harder.
    #   Note: the training loop performs a HARD copy (not soft via ALPHA1).

    WARMUP_EPOCHS = 10000                   # episodes vs random opponent
    SELF_PLAY_UPDATE_LOG = 5000             # episodes between opponent update checks
    SELF_PLAY_UPDATE_WIN_THRESHOLD = 0.50   # min win rate to trigger opponent copy
