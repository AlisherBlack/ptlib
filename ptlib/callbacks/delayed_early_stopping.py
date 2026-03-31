from pytorch_lightning.callbacks import EarlyStopping  # 2.5.6


class DelayedEarlyStopping(EarlyStopping):
    """EarlyStopping that only activates after a specified number of epochs or iterations.

    Args:
        start_after: Number of epochs or iterations to wait before activating.
        count_mode: "epoch" to count epochs, "iter" to count global steps.
        **kwargs: All other arguments are passed to EarlyStopping.
    """

    def __init__(self, start_after: int = 0, count_mode: str = "epoch", **kwargs):
        super().__init__(**kwargs)
        assert count_mode in ("epoch", "iter"), (
            f"count_mode must be 'epoch' or 'iter', got '{count_mode}'"
        )
        self.start_after = start_after
        self.count_mode = count_mode

    def _get_current_count(self, trainer) -> int:
        if self.count_mode == "epoch":
            return trainer.current_epoch
        else:
            return trainer.global_step

    def _should_skip_check(self, trainer) -> bool:
        if self._get_current_count(trainer) < self.start_after:
            return True
        return super()._should_skip_check(trainer)

    def on_validation_end(self, trainer, pl_module):
        if self._get_current_count(trainer) < self.start_after:
            return
        super().on_validation_end(trainer, pl_module)
