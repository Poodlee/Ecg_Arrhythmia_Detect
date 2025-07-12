import torch
from abc import abstractmethod
from numpy import inf
from logger import WandbWriter
import os
import logging
LOG_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, time, data_type):
        
        self.time = time
        
        self.config = config
        verbosity = config['trainer']['verbosity']
        assert verbosity in LOG_LEVELS, f"Invalid verbosity level: {verbosity}. Valid options: {list(LOG_LEVELS.keys())}"

        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(LOG_LEVELS[verbosity])


        self.model = model
        self.criterion = criterion.get_loss()
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        self.data_type = data_type  
        
        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.data_type)

        # setup visualization writer instance                
        self.writer = WandbWriter(config['log_dir'], self.logger, config)
        

        if config['resume'] is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"\n📘 Epoch {epoch} starting...")

            result = self._train_epoch(epoch)
            print(f"✅ Epoch {epoch} training complete. Result: {result}")

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            print("📊 Logging results:")
            for key, value in log.items():
                print(f'    {key:15s}: {value}')
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                print(f"🔍 Monitoring metric: {self.mnt_metric} (mode: {self.mnt_mode})")
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                            (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"⚠️  Warning: Metric '{self.mnt_metric}' is not found. "
                                        f"Model performance monitoring is disabled.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    print(f"💡 Improvement detected: {self.mnt_metric} improved from {self.mnt_best} to {log[self.mnt_metric]}")
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                    print(f"⏸️ No improvement. Count: {not_improved_count}")

                if not_improved_count > self.early_stop:
                    msg = f"🛑 Validation performance didn't improve for {self.early_stop} epochs. Stopping training."
                    print(msg)
                    self.logger.info(msg)
                    break

            if epoch % self.save_period == 0:
                print(f"💾 Saving checkpoint for epoch {epoch} (best: {best})...")
                self._save_checkpoint(epoch, save_best=best)

        print("🏁 Training complete.")

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        save_dir = os.path.join(self.checkpoint_dir, self.time)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(save_dir, 'model_best(epoch{epoch}).pth')

            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
