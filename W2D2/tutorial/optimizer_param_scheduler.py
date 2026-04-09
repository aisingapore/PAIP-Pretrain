# Extracted from megatron/core/optimizer_param_scheduler.py (MegatronLM)
# Standalone version for the WSD scheduler exercise.
#
# The WSD (Warmup-Stable-Decay) implementation has been removed.
# Your task: fill in the TODO blocks in the get_lr() method.

"""Learning rate decay and weight decay incr functions."""
import logging
import math
from typing import Any, Optional, TypedDict

logger = logging.getLogger(__name__)


class ParamGroupOverride(TypedDict):
    """Override values for a parameter group."""

    max_lr: float
    min_lr: float
    start_wd: float
    end_wd: float
    wd_mult: float


def get_canonical_lr_for_logging(param_groups: list[dict]) -> float | None:
    """Return the lr of the first ``default_config=True`` param group."""
    for param_group in param_groups:
        if param_group.get('default_config', False):
            return param_group.get('lr')
    return None


class OptimizerParamScheduler:
    """Anneals learning rate and weight decay.

    This is the core scheduler used by MegatronLM. It supports multiple
    decay styles: constant, linear, cosine, inverse-square-root, and WSD.

    The WSD (Warmup-Stable-Decay) schedule divides training into three phases:
      1. Warmup:  linear ramp from init_lr to max_lr
      2. Stable:  constant at max_lr
      3. Decay:   anneal from max_lr down toward min_lr using a chosen curve

    Args:
        optimizer: the optimizer whose param groups will be updated
        init_lr (float): initial learning rate (start of warmup)
        max_lr (float): peak learning rate
        min_lr (float): minimum learning rate (floor after decay)
        lr_warmup_steps (int): number of warmup steps
        lr_decay_steps (int): total number of training steps
        lr_decay_style (str): one of "constant", "linear", "cosine",
            "inverse-square-root", "WSD"
        start_wd (float): initial weight decay
        end_wd (float): final weight decay
        wd_incr_steps (int): steps over which weight decay increases
        wd_incr_style (str): weight decay increment style
        wsd_decay_steps (int, optional): steps for the WSD decay phase
        lr_wsd_decay_style (str, optional): decay curve within the WSD decay
            phase — "exponential", "linear", "cosine", or "minus_sqrt"
    """

    def __init__(
        self,
        optimizer,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        start_wd: float,
        end_wd: float,
        wd_incr_steps: int,
        wd_incr_style: str,
        use_checkpoint_opt_param_scheduler: Optional[bool] = True,
        override_opt_param_scheduler: Optional[bool] = False,
        wsd_decay_steps: Optional[int] = None,
        lr_wsd_decay_style: Optional[str] = None,
    ) -> None:

        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style
        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, (
                'both override and use-checkpoint are set.'
            )

        # Set the learning rate
        self.step(0)
        logger.info(f"> learning rate decay style: {self.lr_decay_style}")

    def get_wd(self, param_group: Optional[dict] = None) -> float:
        """Weight decay incr functions."""

        if param_group is not None:
            start_wd = param_group.get('start_wd', self.start_wd)
            end_wd = param_group.get('end_wd', self.end_wd)
        else:
            start_wd = self.start_wd
            end_wd = self.end_wd

        if self.num_steps > self.wd_incr_steps:
            return end_wd

        if self.wd_incr_style == 'constant':
            assert start_wd == end_wd
            return end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = end_wd - start_wd

        if self.wd_incr_style == 'linear':
            coeff = incr_ratio
        elif self.wd_incr_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception(f'{self.wd_incr_style} weight decay increment style is not supported.')

        return start_wd + coeff * delta_wd

    def get_lr(self, param_group: dict) -> float:
        """Learning rate decay functions.

        Supports: constant, linear, cosine, inverse-square-root, WSD.

        Args:
            param_group (dict): parameter group from the optimizer.
        """

        max_lr = param_group.get('max_lr', self.max_lr)
        min_lr = param_group.get('min_lr', self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr) * float(self.num_steps) / float(self.lr_warmup_steps)
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == 'constant':
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == 'inverse-square-root':
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        coeff = None
        if self.lr_decay_style == 'linear':
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)

        elif self.lr_decay_style == 'WSD':
            # ================================================================
            # WSD (Warmup-Stable-Decay) schedule
            # ================================================================
            # The stable phase runs from end of warmup until the decay phase
            # begins. The decay phase runs for the last `wsd_decay_steps`
            # steps of training.
            #
            # Reference: MiniCPM (Hu et al., arXiv:2404.06395, 2024)
            # Used by: DeepSeek-V3, Kimi K2, Qwen 3, and many frontier models.
            # ================================================================

            # TODO 1: Calculate when the WSD decay (annealing) phase starts.
            #         The decay phase occupies the LAST wsd_decay_steps of
            #         training. So it starts at:
            #             lr_decay_steps - wsd_decay_steps
            wsd_anneal_start_ = ...  # TODO: replace with the correct expression

            # TODO 2: Determine which phase we're in and compute the
            #         decay ratio if we're in the decay phase.
            if self.num_steps <= wsd_anneal_start_:
                # Stable phase: LR stays at max_lr.
                coeff = ...  # TODO: what coefficient keeps LR at max_lr?
            else:
                # Decay phase: compute progress through the decay.
                wsd_steps = ...       # TODO: how many steps into the decay phase?
                wsd_decay_ratio = ... # TODO: fraction of decay completed (0.0 → 1.0)

                # TODO 3: Implement the four decay styles.
                #
                # Each style maps wsd_decay_ratio ∈ [0, 1] to a coefficient
                # coeff ∈ [0, 1], where coeff=1 means max_lr and coeff=0
                # means min_lr. The final LR is: min_lr + coeff * delta_lr.
                #
                # Hint: look at how "linear" and "cosine" are implemented
                # above (lines ~155-158) for the non-WSD decay styles —
                # the WSD versions follow the same pattern but use
                # wsd_decay_ratio instead of decay_ratio.
                if self.lr_wsd_decay_style == "minus_sqrt":
                    # 1 - sqrt(ratio): fast initial decay, then tapers off.
                    coeff = ...  # TODO
                elif self.lr_wsd_decay_style == "linear":
                    # Straight line from 1 to 0.
                    coeff = ...  # TODO
                elif self.lr_wsd_decay_style == "cosine":
                    # Smooth S-curve from 1 to 0.
                    coeff = ...  # TODO
                elif self.lr_wsd_decay_style == "exponential":
                    # Starts fast, decelerates: (2 * 0.5^ratio) - 1
                    coeff = ...  # TODO

        else:
            raise Exception(f'{self.lr_decay_style} decay style is not supported.')
        assert coeff is not None

        return min_lr + coeff * delta_lr

    def step(self, increment: int) -> None:
        """Set lr for all parameters groups.

        Args:
            increment (int): number of steps to increment
        """
        self.num_steps += increment
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(param_group)
            param_group['weight_decay'] = self.get_wd(param_group) * param_group.get('wd_mult', 1.0)

    def state_dict(self) -> dict:
        """Return the state dict."""
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps,
        }
        return state_dict

    def _check_and_set(self, cls_value: float, sd_value: float, name: str) -> float:
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            logger.info(f" > overriding {name} value to {cls_value}")
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, (
                f'OptimizerParamScheduler: class input value {cls_value} and checkpoint'
                f'value {sd_value} for {name} do not match'
            )

        logger.info(f" > using checkpoint value {sd_value} for {name}")
        return sd_value

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dict."""

        if 'start_lr' in state_dict:
            max_lr_ = state_dict['start_lr']
        else:
            max_lr_ = state_dict['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_, 'learning rate')

        self.min_lr = self._check_and_set(
            self.min_lr, state_dict['min_lr'], 'minimum learning rate'
        )

        if 'warmup_iter' in state_dict:
            lr_warmup_steps_ = state_dict['warmup_iter']
        elif 'warmup_steps' in state_dict:
            lr_warmup_steps_ = state_dict['warmup_steps']
        else:
            lr_warmup_steps_ = state_dict['lr_warmup_steps']
        self.lr_warmup_steps = self._check_and_set(
            self.lr_warmup_steps, lr_warmup_steps_, 'warmup iterations'
        )

        if 'end_iter' in state_dict:
            lr_decay_steps_ = state_dict['end_iter']
        elif 'decay_steps' in state_dict:
            lr_decay_steps_ = state_dict['decay_steps']
        else:
            lr_decay_steps_ = state_dict['lr_decay_steps']
        self.lr_decay_steps = self._check_and_set(
            self.lr_decay_steps, lr_decay_steps_, 'total number of iterations'
        )

        if 'decay_style' in state_dict:
            lr_decay_style_ = state_dict['decay_style']
        else:
            lr_decay_style_ = state_dict['lr_decay_style']
        self.lr_decay_style = self._check_and_set(
            self.lr_decay_style, lr_decay_style_, 'learning rate decay style'
        )

        if 'num_iters' in state_dict:
            num_steps = state_dict['num_iters']
        else:
            num_steps = state_dict['num_steps']
        self.step(increment=num_steps)

        if 'start_wd' in state_dict:
            self.start_wd = self._check_and_set(
                self.start_wd, state_dict['start_wd'], "start weight decay"
            )
            self.end_wd = self._check_and_set(
                self.end_wd, state_dict['end_wd'], "end weight decay"
            )
            self.wd_incr_steps = self._check_and_set(
                self.wd_incr_steps,
                state_dict['wd_incr_steps'],
                "total number of weight decay iterations",
            )
            self.wd_incr_style = self._check_and_set(
                self.wd_incr_style, state_dict['wd_incr_style'], "weight decay incr style"
            )
