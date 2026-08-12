import logging


handler = logging.StreamHandler()          
handler.flush()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[handler]
)
logger = logging.getLogger()
