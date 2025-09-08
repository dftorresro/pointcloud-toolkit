import typer
app = typer.Typer()

@app.command()
def train(config: str = "python/pcseg/cfg/default.yaml"):
    import torch  # lazy import
    from .engine import train as train_mod
    train_mod.main(config)

@app.command()
def view(path: str):
    import open3d as o3d  # lazy import
    from .visualize_open3d import main as viz
    viz(path)

if __name__ == "__main__":
    app()
