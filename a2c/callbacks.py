from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallbackTrain(BaseCallback):
      def __init__(self, verbose=0):
        super(TensorboardCallbackTrain, self).__init__(verbose)

      def _on_training_start(self):
        self._log_freq = 100     # log every 100 calls

        output_formats = self.logger.output_formats

        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

      def _on_step(self) -> bool:
         #if self.n_calls % self._log_freq == 0:
            
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
