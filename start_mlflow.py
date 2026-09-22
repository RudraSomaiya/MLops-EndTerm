"""start mlflow server in-process -- avoids all shell glob expansion."""
import sys

# override argv directly so no shell ever touches the * character
sys.argv = [
    "mlflow",
    "server",
    "--host", "0.0.0.0",
    "--port", "5000",
    "--allowed-hosts", "*",
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "./mlartifacts",
]

from mlflow.cli import cli  # noqa: E402
cli(standalone_mode=True)
