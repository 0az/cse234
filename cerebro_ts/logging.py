import inspect
import logging

from .utils import timestamp

OUTPUT = (logging.WARN + logging.ERROR) // 2


def init_logging() -> None:
    handler = logging.FileHandler(f'/tmp/cerebro-{timestamp()}.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)-5s %(name)s: %(message)s')
    )
    stdout = logging.StreamHandler()
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(
        logging.Formatter('%(levelname)s %(name)s: %(message)s')
    )
    ROOT = logging.getLogger()
    ROOT.addHandler(handler)
    ROOT.addHandler(stdout)
    logging.addLevelName(OUTPUT, 'OUTPUT')
    logging.setLoggerClass(OutputLogger)


# def checkpoint(logger: logging.Logger, label: str) -> None:
#     logger.info()


class OutputLogger(logging.Logger):
    def output(self, msg: str, *args, **kwargs):
        """
        Log 'msg % args' with severity 'OUTPUT'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        if self.isEnabledFor(OUTPUT):
            self._log(OUTPUT, msg, args, **kwargs)

    print = output


def get_logger(name: str = None) -> OutputLogger:
    return logging.getLogger(name)  # type: ignore
