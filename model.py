# ============================================================
# SHIELD FL — model.py
# Combined 1D-CNN BiLSTM + SHIELD LSTM architecture
# Import this in any notebook with: from model import *
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, LayerNormalization, GlobalMaxPooling1D
)
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from scipy.stats import entropy


# ── Combined Model: CNN + BiLSTM + SHIELD LSTM ───────────────

def build_combined_model(n_features, seq_len=5):
    """
    Builds the combined 1D-CNN BiLSTM + SHIELD LSTM model.

    Stage 1 — CNN:         Spatial feature extraction
                           (from Velasquez Restrepo & Luo, 2025)
    Stage 2 — BiLSTM:      Bidirectional temporal modelling
                           (from Velasquez Restrepo & Luo, 2025)
    Stage 3 — SHIELD LSTM: Per-device behavioural profiling
                           (from original SHIELD paper)

    Args:
        n_features: Number of features per timestep
        seq_len:    LSTM sequence length (default 5)
    Returns:
        Uncompiled Keras Model
    """
    reg = tf.keras.regularizers.l2(1e-4)
    inp = Input(shape=(seq_len, n_features), name='input')

    # Stage 1 — CNN
    x = Conv1D(128, kernel_size=3, padding='same',
               activation='relu', kernel_regularizer=reg,
               name='cnn_1')(inp)
    x = BatchNormalization(name='bn_1')(x)
    x = Conv1D(128, kernel_size=3, padding='same',
               activation='relu', kernel_regularizer=reg,
               name='cnn_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(0.3, name='drop_cnn_2')(x)
    x = Conv1D(64, kernel_size=3, padding='same',
               activation='relu', kernel_regularizer=reg,
               name='cnn_3')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Dropout(0.3, name='drop_cnn_3')(x)
    x = LayerNormalization(name='layer_norm')(x)

    # Stage 2 — BiLSTM
    x = Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=reg),
        name='bilstm_1')(x)
    x = Dropout(0.4, name='drop_bilstm_1')(x)
    x = Bidirectional(
        LSTM(32, return_sequences=True, kernel_regularizer=reg),
        name='bilstm_2')(x)
    x = Dropout(0.3, name='drop_bilstm_2')(x)

    # Stage 3 — SHIELD Per-Device LSTM
    x = LSTM(32, return_sequences=False,
             dropout=0.3, recurrent_dropout=0.2,
             kernel_regularizer=reg,
             name='shield_lstm')(x)
    x = BatchNormalization(name='bn_shield')(x)
    x = Dropout(0.3, name='drop_shield')(x)

    # Classification Head
    x = Dense(64, activation='relu', kernel_regularizer=reg,
              kernel_initializer='orthogonal', name='dense_1')(x)
    x = Dropout(0.3, name='drop_dense_1')(x)
    x = Dense(32, activation='relu', kernel_regularizer=reg,
              kernel_initializer='orthogonal', name='dense_2')(x)
    x = Dropout(0.2, name='drop_dense_2')(x)
    out = Dense(1, activation='sigmoid', name='output')(x)

    return Model(inputs=inp, outputs=out, name='SHIELD_CNN_BiLSTM')


# ── Ablation models ───────────────────────────────────────────

def build_cnn_bilstm_only(n_features, seq_len=5):
    """CNN + BiLSTM only — no SHIELD LSTM (Ablation 1)."""
    reg = tf.keras.regularizers.l2(1e-4)
    inp = Input(shape=(seq_len, n_features))
    x = Conv1D(128, 3, padding='same', activation='relu',
               kernel_regularizer=reg)(inp)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='relu',
               kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, 3, padding='same', activation='relu',
               kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=True,
                           kernel_regularizer=reg))(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(32, return_sequences=True,
                           kernel_regularizer=reg))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu', kernel_regularizer=reg,
              kernel_initializer='orthogonal')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out, name='CNN_BiLSTM_Only')


def build_shield_lstm_only(n_features, seq_len=5):
    """SHIELD LSTM only — no CNN/BiLSTM (Ablation 2)."""
    reg = tf.keras.regularizers.l2(1e-4)
    inp = Input(shape=(seq_len, n_features))
    x = LSTM(64, return_sequences=True, dropout=0.3,
             recurrent_dropout=0.2, kernel_regularizer=reg)(inp)
    x = BatchNormalization()(x)
    x = LSTM(32, dropout=0.3, recurrent_dropout=0.2,
             kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu', kernel_regularizer=reg,
              kernel_initializer='orthogonal')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out, name='SHIELD_LSTM_Only')


# ── Compile and callbacks ─────────────────────────────────────

def compile_model(model):
    """Compiles model with exponential decay learning rate."""
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=15000,
        decay_rate=0.95,
        staircase=False
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )
    return model


def get_callbacks():
    """Early stopping — same as Velasquez Restrepo & Luo (2025)."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max',
            min_delta=0.001,
            verbose=0
        )
    ]


# ── Weight extraction and setting ─────────────────────────────

def extract_dense_weights(model):
    """Extracts Dense layer weights for federated sharing."""
    weights, biases = [], []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            w, b = layer.get_weights()
            weights.append(w.copy())
            biases.append(b.copy())
    return weights, biases


def set_dense_weights(model, weights, biases):
    """Applies globally aggregated Dense weights to local model."""
    idx = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights([weights[idx], biases[idx]])
            idx += 1


# ── Evaluation ────────────────────────────────────────────────

def find_threshold(model, X_val, y_val):
    """Finds F1-maximising threshold. Floor 0.30 prevents collapse."""
    y_prob = model.predict(X_val, verbose=0).flatten()
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(y_val, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 3)


def evaluate(model, X, y, threshold=None):
    """Returns full metrics dictionary for given dataset."""
    y_prob = model.predict(X, verbose=0).flatten()
    if threshold is None:
        threshold = find_threshold(model, X, y)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        'accuracy':  round(accuracy_score(y, y_pred), 4),
        'f1':        round(f1_score(y, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y, y_pred, zero_division=0), 4),
        'auc_roc':   round(roc_auc_score(y, y_prob), 4),
        'mcc':       round(matthews_corrcoef(y, y_pred), 4),
        'threshold': threshold,
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn)
    }


# ── DQA scoring ───────────────────────────────────────────────

def calculate_dqa_scores(history, y_train):
    """
    Noise Penalty and Stability Error norm.
    Taken from Velasquez Restrepo & Luo (2025).
    """
    unique, counts = np.unique(y_train, return_counts=True)
    proportions    = counts / len(y_train)
    max_entropy    = np.log(max(len(unique), 2))
    np_score = entropy(proportions) / max_entropy if max_entropy > 0 else 0.0
    train_loss = np.array(history.history['loss'])
    val_loss   = np.array(history.history['val_loss'])
    se_norm    = 1.0 / (1.0 + np.mean(np.abs(val_loss - train_loss)))
    return float(np_score), float(se_norm)


if __name__ == '__main__':
    print('Model functions loaded.')
    print('  build_combined_model()    — main model')
    print('  build_cnn_bilstm_only()   — ablation 1')
    print('  build_shield_lstm_only()  — ablation 2')
