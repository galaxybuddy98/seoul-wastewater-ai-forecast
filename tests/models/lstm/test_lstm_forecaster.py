import numpy as np
import torch

from src.models.lstm.lstm_forecaster import (
    LSTMConfig,
    SeriesLSTM,
    fit_lstm,
    predict_lstm,
)


def test_series_lstm_combines_sequence_and_series_embedding():
    model = SeriesLSTM(input_size=4, num_series=3, hidden_size=8, embedding_size=2)
    sequence = torch.zeros((5, 7, 4), dtype=torch.float32)
    series = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

    output = model(sequence, series)

    assert output.shape == (5,)


def test_fit_lstm_uses_validation_early_stopping_and_returns_finite_predictions():
    rng = np.random.default_rng(42)
    train_x = rng.normal(size=(32, 4, 2)).astype("float32")
    validation_x = rng.normal(size=(12, 4, 2)).astype("float32")
    train_series = np.tile(np.array([0, 1]), 16)
    validation_series = np.tile(np.array([0, 1]), 6)
    train_y = (train_x[:, -1, 0] * 2 + 1).astype("float32")
    validation_y = (validation_x[:, -1, 0] * 2 + 1).astype("float32")
    config = LSTMConfig(hidden_size=8, embedding_size=2, batch_size=16, max_epochs=4, patience=2)

    result = fit_lstm(
        train_x,
        train_series,
        train_y,
        validation_x,
        validation_series,
        validation_y,
        num_series=2,
        config=config,
    )
    predictions = predict_lstm(
        result.model,
        validation_x,
        validation_series,
        result.target_mean,
        result.target_std,
        batch_size=16,
    )

    assert 1 <= result.best_epoch <= 4
    assert len(result.history) <= 4
    assert predictions.shape == (12,)
    assert np.isfinite(predictions).all()
