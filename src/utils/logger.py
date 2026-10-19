import logging
from logging import LoggerAdapter
import datetime
import os
from colorama import init, Fore, Style
# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Default logging configuration
class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors and specific format [time][emulator][file][func][msg]
    """

    # Define colors for different log levels
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }
    TAG_COLORS ={
        # None = normal msg
        "GREEN": Fore.GREEN + Style.BRIGHT,
        "RED": Fore.RED + Style.BRIGHT,
        "YELLOW": Fore.YELLOW + Style.BRIGHT,
        "PURPLE": Fore.MAGENTA + Style.BRIGHT, # system messages
    }
    def regex_tag(self, msg: str):
        # extract one [TAG] from message if present
        if msg.startswith('['):
            end_idx = msg.find(']')
            if end_idx != -1:
                tag = msg[1:end_idx]
                rest_msg = msg[end_idx+1:].strip()
                return tag, rest_msg # type: ignore
        return None, msg # type: ignore

    def format(self, record):
        """
        Format the log record with timestamp, emulator, file, function, and colored message.

        Args:
            record: The log record to format.

        Returns:
            str: Formatted log string.
        """
        msg = record.getMessage()
        tag, res_msg = self.regex_tag(msg)
        # Get the color for the log level
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        tag_color= self.TAG_COLORS.get(tag, None)

        if record.levelno != logging.INFO  and tag_color:
            print("="*40,"\n\nSHOULD NOT USE TAG COLOR FOR NON-INFO LEVELS\n", f"at: [{record.filename}][{record.funcName}]:[{record.lineno}]\n\n" , "="*40) 
        
        if tag_color:
            color = tag_color # Override color if tag found
            msg = res_msg # Use the message without the tag

        # Format: [YYYY-MM-DD HH:MM:SS][emulator][filename][function][message]
        #log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_format = f"{record.filename}][{record.funcName}] - {color}{msg}{Style.RESET_ALL}"
        return log_format


def setup_logger(
    log_file,
    log_level=logging.WARNING,
) -> LoggerAdapter:
    """
    Sets up the logger with file and console handlers, using the custom colored formatter.
    
    Creates a timestamped subdirectory for each run (e.g., Logs/20251031_143022/).

    Args:
        log_file (str): Name of the log file (e.g., 'trainer.log')
        log_level (int): Logging level (e.g., logging.INFO).
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Construct full log file path
    log_file_path = log_file
    # Create logger
    logger = logging.getLogger(f"Logger_{log_file_path}")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)  # Set console to WARNING level
        console_handler.setFormatter(ColoredFormatter())

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return LoggerAdapter(logger)
