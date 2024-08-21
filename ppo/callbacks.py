from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat, HParam
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallbackTrain(BaseCallback):
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class TensorboardCallbackTest(BaseCallback):
      def __init__(self, log_dir, verbose=0):
         super(TensorboardCallbackTest, self).__init__(verbose)
         self.writer = SummaryWriter(log_dir)

      def _on_step(self) -> bool:
         if self.locals["dones"]:
            self.writer.add_scalar('Test/Step Reward', self.locals["rewards"][0], self.num_timesteps)
         return True

      def log_episode_reward(self, episode_reward, episode):
         self.writer.add_scalar('Test/Episode Reward', episode_reward, episode)

      def log_final_metrics(self, avg_reward, std_reward, best_reward):
         self.writer.add_text('Avg Reward', f"{avg_reward:.2f}", 0)
         self.writer.add_text('Std Reward', f"{std_reward:.2f}", 0)
         self.writer.add_text('Best Reward', f"{best_reward:.2f}", 0)
