# ============================================================
# SHIELD FL — config.py
# All settings for the entire experiment.
# Change values here only — everything else reads from this file.
# ============================================================

# ── Dataset ──────────────────────────────────────────────────
# Change this to your TON_IoT file location
# Google Colab example:
#   TON_IOT_PATH = '/content/drive/MyDrive/TON_IoT_Network.csv'
TON_IOT_PATH = '/content/TON_IoT_Network.csv'
DATA_SAVE_DIR = 'shield_fl_data'     # Folder to save prepared data

# ── Federated learning ────────────────────────────────────────
NUM_CLIENTS  = 3     # Healthcare organisations
NUM_ROUNDS   = 10     # Training rounds
SEQUENCE_LEN = 5     # Consecutive flows per LSTM input
BATCH_SIZE   = 512
MAX_EPOCHS   = 80
SEED         = 42

# ── Adversarial settings ──────────────────────────────────────
POISON_CLIENT   = 3      # Which client is the attacker (1, 2, or 3)
POISON_STRENGTH = 2.0    # Noise multiplier — higher = stronger attack
DP_NOISE_SCALE  = 0.01   # Gaussian noise for differential privacy

if __name__ == '__main__':
    print('Configuration loaded.')
    print(f'  Dataset : {TON_IOT_PATH}')
    print(f'  Clients : {NUM_CLIENTS}')
    print(f'  Rounds  : {NUM_ROUNDS}')
