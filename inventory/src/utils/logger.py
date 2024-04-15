import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision


class Logger(object):
    def __init__(self, log_dir, log_name, log_hist=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:    # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir,
                log_name + '_' + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def list_of_image(self, tag, images, step):
        """Log scalar variables."""
        grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, grid, step)


