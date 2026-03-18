"""pytorchexample: A Flower / PyTorch app."""
import json
import torch
from datetime import datetime
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from pytorchexample.task import Net, load_centralized_dataset, test

# Directorio de outputs global
output_dir = None
history = []

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    global output_dir, history

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Crear directorio de outputs
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(f"outputs/{run_time}")
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    print(f"\nGuardando resultados en: {output_dir}")

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    global output_dir, history

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    # Guardar métricas
    if output_dir is not None:
        history.append({"round": server_round, "accuracy": test_acc, "loss": test_loss})
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})