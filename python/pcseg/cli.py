import typer; from pcseg.engine.train import train_loop
app = typer.Typer(add_completion=False)

@app.command()
def train(cfg: str = "python/pcseg/cfg/default.yaml"):
    """Train point-cloud segmentation model."""
    train_loop(cfg)

if __name__ == "__main__": app()
