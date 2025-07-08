# wandb logging 
import wandb
import torch
import os

class WandbWriter:
    def __init__(self, log_dir, logger, config, project_name="TMAT Model Test"):
        """
        WandbWriter for logging metrics, images, and models to Weights & Biases.
        
        Args:
            log_dir (str): Directory to save model checkpoints.
            logger: Logger object for debugging (e.g., logging.Logger).
            config (dict): Configuration dictionary for wandb.init and experiment settings.
            project_name (str): W&B project name (default: 'ppp6131-yonsei-university').
        """
        self.log_dir = log_dir
        self.logger = logger
        self.config = config

        self._scope_steps = {}

        # Initialize wandb
        wandb.login()
        wandb.init(project=project_name, config=config)
        wandb.run.name = self._map_dict_to_str(config)
        self.logger.info(f"W&B run initialized: {wandb.run.name}")

    def _map_dict_to_str(self, config):
        """Convert config dict to a string for run name."""
        parts = []
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    parts.append(f"{key}_{subkey}={subvalue}")
            else:
                parts.append(f"{key}={value}")
        return "_".join(parts)

    def set_step(self, step, scope=None):
        if scope is None:
            self._current_step = step
        else:
            self._scope_steps[scope] = step

    def get_step(self, scope=None):
        return self._scope_steps.get(scope, self._current_step)

    def add_scalar(self, key, value, step=None):
        wandb.log({key: value}, step=step if step is not None else self._current_step)

    def add_image(self, key, image, caption=None, step=None):
        wandb.log({key: wandb.Image(image, caption=caption)}, step=step if step is not None else self._current_step)

    def add_histogram(self, name, values, step=None):
        """
        Logs a histogram. wandb.histogram() expects a 1D tensor or numpy array.
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        wandb.log({name: wandb.Histogram(values)}, step=step if step is not None else self._current_step)

    def wandb_donot_watch(self, model, criterion=None, log="none"):
        wandb.watch(model, criterion, log="none")

    def watch(self, model, criterion=None, log="all"):
        wandb.watch(model, criterion, log=log)

    def save_model(self, model, path="Best_Model.pth"):
        save_path = os.path.join(self.log_dir, path)
        torch.save(model.state_dict(), save_path)
        wandb.save(save_path)
        self.logger.info(f"Saved model to {save_path} and uploaded to W&B")

    def log(self, metrics, step=None):
        wandb.log(metrics, step=step if step is not None else self._current_step)

    def finish(self):
        wandb.finish()
        
        