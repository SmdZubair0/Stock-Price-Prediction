import logging
from pathlib import Path
import datetime as dt


def create_log_path(module_name : str) -> Path:
    """
        This function creates a folder to store logs for the given module_name (where this function is called)
    """

    # access root directory
    root_path = Path(__file__).parent.parent

    # create a logs folder
    log_dir = root_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # current date
    current_date = dt.datetime.today().strftime("%d-%m-%y")

    # create a folder based on in which file you are making this logs
    log_set = log_dir / module_name
    log_set.mkdir(exist_ok=True, parents=True)

    # create current log file
    log_file = log_set / (current_date + ".log")

    return log_file



class CustomLogger:
    def __init__(self, logger_name, logger_filepath) -> None:
        
        self.__log_filepath = logger_filepath  # store the file path
        self.__logger = logging.getLogger(logger_name)  # create a logger object with given name

        # this will create a filehandler for logging files with append mode
        fileHandler = logging.FileHandler(self.__log_filepath,
                                          mode='a')
        
        self.__logger.addHandler(fileHandler) # add this file handler to logger object

        # set format for logs and time
        log_format = "%(asctime)s : %(levelname)s : %(message)s"  # stores as time, levelname, message
        time_format = "%d-%h-%y %H-%M-%S"
        # create the formatter
        formatter = logging.Formatter(log_format,
                          datefmt=time_format)
        
        fileHandler.setFormatter(formatter)  # add this formater to filehandler


    def get_logger(self):
        """
        Return logger object
        """
        return self.__logger
    
    def get_log_filepath(self):
        """
        Return log file path
        """
        return self.__log_filepath
    
    def set_log_level(self, log_level = logging.INFO):
        """
        Sets the log level for the logger.

        Parameters:
        - level : Logging level (e.g., logging.DEBUG, logging.INFO).
        """
        logger = self.get_logger()
        logger.setLevel(level=log_level)
        
    def save_logs(self,msg,log_level='info'):
        """
        Saves logs to the specified log file with the given message and log level.

        Parameters:
        - msg (str): Log message.
        - log_level (str): Log level ('debug', 'info', 'warning', 'error', 'exception', 'critical').
        """
        # get the logger
        logger = self.get_logger()
        # save the logs to the file using the given message
        if log_level == 'debug':
            logger.debug(msg=msg)
        elif log_level == 'info':
            logger.info(msg=msg)
        elif log_level == 'warning':
            logger.warning(msg=msg)
        elif log_level == 'error':
            logger.error(msg=msg)
        elif log_level == 'exception':
            logger.exception(msg=msg)
        elif log_level == 'critical':
            logger.critical(msg=msg)



if __name__ == "__main__":
    pass