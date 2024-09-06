import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import yaml
from models import FluxModel
from ui_manager import create_latent_walk_interface
import asyncio
import sys
import traceback

def load_config(config_path: str) -> dict:
    try:
        with open(config_path) as yaml_file:
            return yaml.safe_load(yaml_file)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="Run the Latent Space exploration game.")
    parser.add_argument('--config', type=str, default="./configs/latent_space_game.yaml",
                        help='Path to the game configuration file.')
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        flux_model = FluxModel(cfg)
        await create_latent_walk_interface(args.config, cfg, flux_model)
    except Exception as e:
        print(f"An error occurred in main: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGame terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)