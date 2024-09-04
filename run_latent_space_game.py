import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import yaml
from models import FluxModel
from ui_manager import create_latent_walk_interface

def main():
    parser = argparse.ArgumentParser(description="Run the Latent Space exploration game.")
    parser.add_argument('--config', type=str, default="./configs/latent_space_game.yaml",
                        help='Path to the game configuration file.')
    args = parser.parse_args()

    with open(args.config) as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    flux_model = FluxModel(cfg)
    create_latent_walk_interface(args.config, cfg, flux_model)

if __name__ == "__main__":
    main()