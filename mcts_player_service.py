import logging
import os
from flask import request
from jass.service.player_service_app import PlayerServiceApp
from mcts_agent import AgentMCTSTrumpSchieber

def configure_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('player_service')
    return logger

logger = configure_logging()

def log_request_info():
    """Log incoming request details."""
    logger.info(f"Incoming request: {request.method} {request.url}")
    if request.data:
        logger.debug(f"Request Data: {request.data}")

def create_app() -> PlayerServiceApp:
    """Create and configure the Flask app."""
    app = PlayerServiceApp('player_service')
    
    app.add_player('mcts', AgentMCTSTrumpSchieber())

    app.before_request(log_request_info)

    return app

if __name__ == '__main__':
    app = create_app()
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8888))
    
    app.run(host=host, port=port)
