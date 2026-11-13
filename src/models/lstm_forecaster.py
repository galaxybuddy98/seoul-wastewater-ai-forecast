"""계통 임베딩을 포함한 CPU 경량 LSTM 회귀 모델."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED = 42


@dataclass(frozen=True)
class LSTMConfig:
    hidden_size: int = 32
    embedding_size: int = 4
    batch_size: int = 512
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    seed: int = RANDOM_SEED


@dataclass
class LSTMFitResult:
    model: "SeriesLSTM"
    target_mean: float
    target_std: float
    best_epoch: int
    history: list[dict[str, float]]
    training_seconds: float
    config: dict[str, object]


class SeriesLSTM(nn.Module):
    """시퀀스 마지막 은닉상태와 Train 계통 임베딩을 결합한다."""

    def __init__(
        self,
        input_size: int,
        num_series: int,
        hidden_size: int = 32,
        embedding_size: int = 4,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.series_embedding = nn.Embedding(num_series, embedding_size)
        self.output = nn.Linear(hidden_size + embedding_size, 1)

    def forward(self, sequence: torch.Tensor, series_code: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        combined = torch.cat([hidden[-1], self.series_embedding(series_code)], dim=1)
        return self.output(combined).squeeze(1)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def predict_lstm(
    model: SeriesLSTM,
    values: np.ndarray,
    series_codes: np.ndarray,
    target_mean: float,
    target_std: float,
    batch_size: int = 512,
) -> np.ndarray:
    """저장된 목표 스케일을 되돌린 LSTM 예측을 반환한다."""
    dataset = TensorDataset(
        torch.as_tensor(values, dtype=torch.float32),
        torch.as_tensor(series_codes, dtype=torch.long),
    )
    model.eval()
    parts = []
    with torch.no_grad():
        for sequence, series in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            parts.append(model(sequence, series).cpu().numpy())
    scaled = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return scaled * target_std + target_mean


def fit_lstm(
    train_x: np.ndarray,
    train_series: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_series: np.ndarray,
    validation_y: np.ndarray,
    num_series: int,
    config: LSTMConfig | None = None,
) -> LSTMFitResult:
    """Validation 실제 단위 MAE 조기종료로 단일 목표 LSTM을 적합한다."""
    config = config or LSTMConfig()
    _seed_everything(config.seed)
    target_mean = float(np.mean(train_y))
    target_std = float(np.std(train_y))
    if target_std == 0:
        target_std = 1.0
    scaled_y = ((np.asarray(train_y) - target_mean) / target_std).astype(np.float32)
    dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_series, dtype=torch.long),
        torch.as_tensor(scaled_y, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = SeriesLSTM(
        input_size=train_x.shape[2],
        num_series=num_series,
        hidden_size=config.hidden_size,
        embedding_size=config.embedding_size,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.L1Loss()
    best_state = deepcopy(model.state_dict())
    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    started = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for sequence, series, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(sequence, series), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(sequence)
            row_count += len(sequence)
        validation_prediction = predict_lstm(
            model,
            validation_x,
            validation_series,
            target_mean,
            target_std,
            batch_size=config.batch_size,
        )
        validation_mae = float(np.mean(np.abs(validation_y - validation_prediction)))
        history.append(
            {
                "epoch": float(epoch),
                "train_scaled_mae": loss_sum / row_count,
                "validation_mae": validation_mae,
            }
        )
        if validation_mae < best_mae - 1e-8:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.load_state_dict(best_state)
    return LSTMFitResult(
        model=model,
        target_mean=target_mean,
        target_std=target_std,
        best_epoch=best_epoch,
        history=history,
        training_seconds=perf_counter() - started,
        config=asdict(config),
    )
