import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from wtb._llm_response_generation import get_args, main
from wtb.constant import DOTENV_PATH
from dotenv import load_dotenv

                                                                                         
                                                                   
                                                
if __name__ == "__main__":
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)                      

    main(get_args())

    """
    
     python wtb/openfunctions_evaluation.py --model Kimi-K2.6 --num-threads 1
     
     """
