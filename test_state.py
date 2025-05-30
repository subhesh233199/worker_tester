from threading import Lock
import logging
from TM_crew_setup import validate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Shared state for thread-safe data sharing
class SharedState:
    """
    Thread-safe shared state container for the application.
    
    Attributes:
        metrics (dict): Stores processed metrics data
        parsed_metrics (dict): Stores metrics parsed from markdown
        report_parts (dict): Stores different sections of the generated report
        lock (Lock): Thread lock for general operations
        visualization_ready (bool): Flag indicating visualization status
        viz_lock (Lock): Thread lock for visualization operations
    """
    def __init__(self):
        self.metrics = None
        self.parsed_metrics = None
        self.report_parts = {}
        self.lock = Lock()
        self.visualization_ready = False
        self.viz_lock = Lock()

    def update_metrics(self, new_metrics: dict) -> bool:
        """
        Update parsed metrics and validate them, resetting visualization status.

        Args:
            new_metrics (dict): New metrics data to store and validate

        Returns:
            bool: True if metrics are valid and updated, False otherwise
        """
        with self.lock:
            try:
                if not new_metrics or not isinstance(new_metrics, dict):
                    logger.error("Invalid metrics data provided")
                    return False

                # Validate metrics
                if not validate_metrics(new_metrics):
                    logger.error("Metrics validation failed")
                    return False

                # Store parsed metrics
                self.parsed_metrics = new_metrics
                logger.info("Parsed metrics updated successfully")

                # Update processed metrics
                self.metrics = new_metrics
                logger.info("Processed metrics updated successfully")

                # Reset visualization status
                self.visualization_ready = False
                logger.info("Visualization status reset")

                return True
            except Exception as e:
                logger.error(f"Error updating metrics: {str(e)}")
                return False

shared_state = SharedState()
