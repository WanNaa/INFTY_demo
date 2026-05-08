import numpy as np
import torch


class FeedbackDegradationController:
    def __init__(self, args):
        seed = args["seed"][0] if isinstance(args.get("seed"), (list, tuple)) else int(args.get("seed", 0))
        self.rng = np.random.RandomState(seed + 20260326)

        self.enabled = bool(args.get("feedback_stress_enabled", False))
        self.mode = args.get("feedback_stress_mode", "none")
        self.sparse_mode = args.get("feedback_sparse_mode", "steps")
        self.keep_prob = float(args.get("feedback_keep_prob", 1.0))
        self.label_keep_prob = float(args.get("feedback_label_keep_prob", 1.0))
        self.noise_type = args.get("feedback_noise_type", "none")
        self.noise_level = float(args.get("feedback_noise_level", 0.0))
        self.noise_application = args.get("feedback_noise_application", "multiplicative")
        self.level = float(args.get("feedback_stress_level", 0.0))

        self.step_index = 0

    def reset_task(self, task_id):
        self.step_index = 0
        self.rng.seed(self.rng.randint(0, 2**31 - 1) + int(task_id) * 9973)

    def begin_batch(self, batch_size, device):
        self.step_index += 1

        step_has_feedback = True
        sample_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        label_mask = torch.ones(batch_size, device=device, dtype=torch.bool)

        if not self.enabled or self.mode == "none":
            return {
                "step_has_feedback": step_has_feedback,
                "sample_mask": sample_mask,
                "label_mask": label_mask,
            }

        if self.mode == "sparse_feedback":
            keep_prob = min(max(self.keep_prob, 0.0), 1.0)
            if self.sparse_mode == "steps":
                step_has_feedback = bool(self.rng.rand() < keep_prob)
            elif self.sparse_mode == "samples":
                sample_mask_np = self.rng.rand(batch_size) < keep_prob
                sample_mask = torch.as_tensor(sample_mask_np, device=device, dtype=torch.bool)
            else:
                raise ValueError(f"Unsupported feedback_sparse_mode: {self.sparse_mode}")

        if self.mode == "reduced_supervision":
            keep_prob = min(max(self.label_keep_prob, 0.0), 1.0)
            label_mask_np = self.rng.rand(batch_size) < keep_prob
            label_mask = torch.as_tensor(label_mask_np, device=device, dtype=torch.bool)

        return {
            "step_has_feedback": step_has_feedback,
            "sample_mask": sample_mask,
            "label_mask": label_mask,
        }

    def apply_to_per_sample_loss(self, per_sample_loss, context):
        if not self.enabled or self.mode == "none":
            return per_sample_loss.mean()

        if not context["step_has_feedback"]:
            return per_sample_loss.sum() * 0.0

        effective_mask = context["sample_mask"] & context["label_mask"]
        if not bool(effective_mask.any()):
            return per_sample_loss.sum() * 0.0

        masked_loss = per_sample_loss[effective_mask]
        loss = masked_loss.mean()
        return self.apply_noise(loss)

    def apply_noise(self, loss):
        if not self.enabled or self.mode != "noisy_feedback":
            return loss
        if self.noise_type == "none" or self.noise_level <= 0:
            return loss

        if self.noise_type == "gaussian":
            noise = float(self.rng.normal(0.0, self.noise_level))
        elif self.noise_type == "uniform":
            noise = float(self.rng.uniform(-self.noise_level, self.noise_level))
        else:
            raise ValueError(f"Unsupported feedback_noise_type: {self.noise_type}")

        if self.noise_application != "multiplicative":
            raise ValueError(f"Unsupported feedback_noise_application: {self.noise_application}")

        scale = max(0.0, 1.0 + noise)
        return loss * loss.new_tensor(scale)
