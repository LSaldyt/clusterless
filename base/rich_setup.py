''' Setup pretty printing, logging, and tracebacks for this project '''
import logging
from rich.logging import RichHandler
from rich.traceback import install
install(show_locals=False)
from rich.pretty import pprint
from rich import print
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")
