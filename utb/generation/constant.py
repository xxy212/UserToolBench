from pathlib import Path

                                                                                     
DOTENV_PATH = ".env"

                                                 
script_dir = Path(__file__).parent
DOTENV_PATH = (script_dir / DOTENV_PATH).resolve()
