    # trainer.py → 負責訓練、validation、token exposure logging、checkpoint
    # 1. forward
    # 2. loss
    # 3. backward
    # 4. optimizer step
    # 5. 用 attention_mask 計算 batch token 數
    # 6. 根據 language 分別累加 EN / NL / ZH raw tokens
    # 7. 套用 Byte Premium
    # 8. 記錄 CSV
    # 9. 如果達到 token milestone，存 checkpoint
    # 10. 如果超過 exposure limit，停止
import os
import csv
import math
import logging
import json
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


BYTE_PREMIUM = {
    "eng": 1.0,
    "nld": 1.0516,
    "zho": 0.9894,
}


class TokenExposureTracker:
    def __init__(self):
        self.raw_seen_tokens_eng = 0
        self.raw_seen_tokens_nld = 0
        self.raw_seen_tokens_zho = 0

    def update(self, attention_mask, languages):
        """
        attention_mask: Tensor [batch_size, seq_len]
        languages: list[str], e.g. ["eng", "nld", "zho"]
        """
        tokens_per_sample = attention_mask.sum(dim=1).detach().cpu().tolist()

        for tokens, lang in zip(tokens_per_sample, languages):
            tokens = int(tokens)

            if lang == "eng":
                self.raw_seen_tokens_eng += tokens
            elif lang == "nld":
                self.raw_seen_tokens_nld += tokens
            elif lang == "zho":
                self.raw_seen_tokens_zho += tokens
            else:
                raise ValueError(f"Unknown language label: {lang}")
            
    def update_from_language_ids(self, attention_mask, language_ids):
        """
        For: wrapped packing
        Update token exposure using per-token language ids.

        language_ids:
        0 = eng
        1 = nld
        2 = zho
        -100 = padding / ignored position
        """

        valid_mask = attention_mask.bool()

        eng_tokens = ((language_ids == 0) & valid_mask).sum().item()
        nld_tokens = ((language_ids == 1) & valid_mask).sum().item()
        zho_tokens = ((language_ids == 2) & valid_mask).sum().item()

        self.raw_seen_tokens_eng += eng_tokens
        self.raw_seen_tokens_nld += nld_tokens
        self.raw_seen_tokens_zho += zho_tokens

    @property
    def raw_seen_tokens_total(self):
        return (
            self.raw_seen_tokens_eng
            + self.raw_seen_tokens_nld
            + self.raw_seen_tokens_zho
        )

    @property
    def adjusted_seen_tokens_eng(self):
        return self.raw_seen_tokens_eng * BYTE_PREMIUM["eng"]

    @property
    def adjusted_seen_tokens_nld(self):
        return self.raw_seen_tokens_nld * BYTE_PREMIUM["nld"]

    @property
    def adjusted_seen_tokens_zho(self):
        return self.raw_seen_tokens_zho * BYTE_PREMIUM["zho"]

    @property
    def adjusted_seen_tokens_total(self):
        return (
            self.adjusted_seen_tokens_eng
            + self.adjusted_seen_tokens_nld
            + self.adjusted_seen_tokens_zho
        )


def compute_adjusted_token_delta(attention_mask, languages=None, language_ids=None):
    if language_ids is not None:
        valid_mask = attention_mask.bool()
        eng_tokens = ((language_ids == 0) & valid_mask).sum().item()
        nld_tokens = ((language_ids == 1) & valid_mask).sum().item()
        zho_tokens = ((language_ids == 2) & valid_mask).sum().item()
    else:
        if languages is None:
            raise ValueError("languages or language_ids must be provided.")

        eng_tokens = 0
        nld_tokens = 0
        zho_tokens = 0
        tokens_per_sample = attention_mask.sum(dim=1).detach().cpu().tolist()

        for tokens, lang in zip(tokens_per_sample, languages):
            tokens = int(tokens)
            if lang == "eng":
                eng_tokens += tokens
            elif lang == "nld":
                nld_tokens += tokens
            elif lang == "zho":
                zho_tokens += tokens
            else:
                raise ValueError(f"Unknown language label: {lang}")

    return (
        eng_tokens * BYTE_PREMIUM["eng"]
        + nld_tokens * BYTE_PREMIUM["nld"]
        + zho_tokens * BYTE_PREMIUM["zho"]
    )


def get_next_checkpoint_target(current_adjusted_tokens, checkpoint_interval_tokens=None):
    """
    Official-style checkpoint schedule:
    0–10M: every 1M
    10M–100M: every 10M
    100M–1B: every 100M
    """
    if checkpoint_interval_tokens is not None:
        interval = checkpoint_interval_tokens
    elif current_adjusted_tokens < 10_000_000:
        interval = 1_000_000
    elif current_adjusted_tokens < 100_000_000:
        interval = 10_000_000
    else:
        interval = 100_000_000

    return math.ceil((current_adjusted_tokens + 1) / interval) * interval


def format_checkpoint_name(token_count):
    if token_count < 1_000_000:
        return f"step_{int(token_count)}"
    if token_count < 1_000_000_000:
        return f"step_{int(token_count / 1_000_000)}M"
    return f"step_{int(token_count / 1_000_000_000)}B"


def resave_hf_fast_tokenizer(checkpoint_path, model_max_length=512):
    tokenizer_json_path = os.path.join(checkpoint_path, "tokenizer.json")

    if not os.path.exists(tokenizer_json_path):
        raise FileNotFoundError(
            f"tokenizer.json not found after tokenizer.save_pretrained: "
            f"{tokenizer_json_path}"
        )

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_path,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<s>",
        sep_token="</s>",
        model_max_length=model_max_length,
    )
    hf_tokenizer.save_pretrained(checkpoint_path)

    tokenizer_config_path = os.path.join(checkpoint_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)

        tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
        tokenizer_config["model_max_length"] = model_max_length

        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
            f.write("\n")


def infer_model_max_length(model, default=512):
    config = getattr(model, "config", None)
    if config is None:
        return default

    for attr in ("max_position_embeddings", "n_positions"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)

    return default


def save_checkpoint(model, tokenizer, output_dir, checkpoint_name):
    checkpoint_path = os.path.join(output_dir, "checkpoints", checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    model.save_pretrained(checkpoint_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_path)
        resave_hf_fast_tokenizer(
            checkpoint_path,
            model_max_length=infer_model_max_length(model),
        )

    return checkpoint_path


def evaluate(model, val_loader, device, max_val_steps=None):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            batch.pop("language", None)
            batch.pop("language_ids", None)

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            steps += 1

            if max_val_steps is not None and steps >= max_val_steps:
                break

    model.train()

    if steps == 0:
        return None

    return total_loss / steps


def train_model(
    model,
    train_loader,
    val_loader,
    tokenizer,
    config,
):
    logger = logging.getLogger(__name__)

    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "training_log.csv")

    learning_rate = float(config.get("learning_rate", 5e-4))
    max_epochs = int(config.get("max_epochs", 1))
    max_steps = config.get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None
    max_val_steps = int(config.get("max_val_steps", 20))
    max_adjusted_token_exposure = float(
        config.get("max_adjusted_token_exposure", 1_000_000_000)
    )
    checkpoint_interval_tokens = config.get("checkpoint_interval_tokens", None)
    checkpoint_interval_tokens = (
        int(checkpoint_interval_tokens)
        if checkpoint_interval_tokens is not None
        else None
    )
    save_intermediate_checkpoints = bool(
        config.get("save_intermediate_checkpoints", True)
    )

    if max_epochs > 10:
        raise ValueError("BabyLM rule: max_epochs must be <= 10")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info(f"Using device: {device}")
    logger.info(
        f"Save intermediate checkpoints: {save_intermediate_checkpoints}"
    )

    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    tracker = TokenExposureTracker()

    global_step = 0
    next_checkpoint_target = None
    if save_intermediate_checkpoints:
        next_checkpoint_target = get_next_checkpoint_target(
            tracker.adjusted_seen_tokens_total,
            checkpoint_interval_tokens,
        )

    fieldnames = [
        "epoch",
        "global_step",
        "train_loss",
        "validation_loss",
        "raw_seen_tokens_total",
        "raw_seen_tokens_eng",
        "raw_seen_tokens_nld",
        "raw_seen_tokens_zho",
        "adjusted_seen_tokens_eng",
        "adjusted_seen_tokens_nld",
        "adjusted_seen_tokens_zho",
        "adjusted_seen_tokens_total",
        "checkpoint_path",
    ]
    last_train_loss = None

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def write_final_checkpoint_row(
            epoch,
            global_step,
            train_loss,
            checkpoint_name="final_model",
        ):
            final_validation_loss_value = (
                evaluate(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    max_val_steps=max_val_steps,
                )
                if val_loader is not None
                else None
            )
            final_validation_loss = (
                f"{final_validation_loss_value:.6f}"
                if final_validation_loss_value is not None
                else ""
            )
            final_path = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                checkpoint_name=checkpoint_name,
            )
            writer.writerow({
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": f"{train_loss:.6f}",
                "validation_loss": final_validation_loss,
                "raw_seen_tokens_total": tracker.raw_seen_tokens_total,
                "raw_seen_tokens_eng": tracker.raw_seen_tokens_eng,
                "raw_seen_tokens_nld": tracker.raw_seen_tokens_nld,
                "raw_seen_tokens_zho": tracker.raw_seen_tokens_zho,
                "adjusted_seen_tokens_eng": f"{tracker.adjusted_seen_tokens_eng:.2f}",
                "adjusted_seen_tokens_nld": f"{tracker.adjusted_seen_tokens_nld:.2f}",
                "adjusted_seen_tokens_zho": f"{tracker.adjusted_seen_tokens_zho:.2f}",
                "adjusted_seen_tokens_total": f"{tracker.adjusted_seen_tokens_total:.2f}",
                "checkpoint_path": final_path,
            })
            f.flush()
            logger.info(f"Final validation loss: {final_validation_loss}")
            return final_path

        for epoch in range(1, max_epochs + 1):
            logger.info(f"Starting epoch {epoch}/{max_epochs}")

            progress = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch in progress:
                languages = batch.pop("language", None)
                language_ids = batch.pop("language_ids", None)

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }

                if language_ids is not None:
                    language_ids = language_ids.to(device)

                batch_adjusted_delta = compute_adjusted_token_delta(
                    attention_mask=batch["attention_mask"],
                    languages=languages,
                    language_ids=language_ids,
                )
                if (
                    tracker.adjusted_seen_tokens_total + batch_adjusted_delta
                    > max_adjusted_token_exposure
                ):
                    logger.info(
                        "Stopping before next batch: adjusted token exposure "
                        f"would exceed {max_adjusted_token_exposure:,} "
                        f"(current={tracker.adjusted_seen_tokens_total:.2f}, "
                        f"batch_delta={batch_adjusted_delta:.2f})"
                    )
                    if last_train_loss is not None:
                        final_path = write_final_checkpoint_row(
                            epoch=epoch,
                            global_step=global_step,
                            train_loss=last_train_loss,
                        )
                    else:
                        final_path = save_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=output_dir,
                            checkpoint_name="final_model",
                        )
                    logger.info(f"Final checkpoint saved to {final_path}")
                    return

                optimizer.zero_grad()

                outputs = model(**batch)
                loss = outputs.loss
                last_train_loss = loss.item()

                loss.backward()
                optimizer.step()

                global_step += 1

                # Count raw seen tokens by language
                if language_ids is not None:
                    tracker.update_from_language_ids(
                        attention_mask=batch["attention_mask"],
                        language_ids=language_ids,
                    )
                else:
                    tracker.update(
                        attention_mask=batch["attention_mask"],
                        languages=languages,
                    )

                adjusted_total = tracker.adjusted_seen_tokens_total

                checkpoint_path = ""

                # Save checkpoint by adjusted token milestones
                if (
                    save_intermediate_checkpoints
                    and adjusted_total >= next_checkpoint_target
                ):
                    checkpoint_name = format_checkpoint_name(next_checkpoint_target)
                    checkpoint_path = save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=output_dir,
                        checkpoint_name=checkpoint_name,
                    )

                    logger.info(
                        f"Saved checkpoint at adjusted token target "
                        f"{next_checkpoint_target:,}: {checkpoint_path}"
                    )

                    next_checkpoint_target = get_next_checkpoint_target(
                        adjusted_total,
                        checkpoint_interval_tokens,
                    )

                # Validation
                validation_loss = ""
                if val_loader is not None and global_step % config.get("eval_every_steps", 20) == 0:
                    validation_loss_value = evaluate(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        max_val_steps=max_val_steps,
                    )
                    validation_loss = (
                        f"{validation_loss_value:.6f}"
                        if validation_loss_value is not None
                        else ""
                    )

                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": f"{loss.item():.6f}",
                    "validation_loss": validation_loss,
                    "raw_seen_tokens_total": tracker.raw_seen_tokens_total,
                    "raw_seen_tokens_eng": tracker.raw_seen_tokens_eng,
                    "raw_seen_tokens_nld": tracker.raw_seen_tokens_nld,
                    "raw_seen_tokens_zho": tracker.raw_seen_tokens_zho,
                    "adjusted_seen_tokens_eng": f"{tracker.adjusted_seen_tokens_eng:.2f}",
                    "adjusted_seen_tokens_nld": f"{tracker.adjusted_seen_tokens_nld:.2f}",
                    "adjusted_seen_tokens_zho": f"{tracker.adjusted_seen_tokens_zho:.2f}",
                    "adjusted_seen_tokens_total": f"{tracker.adjusted_seen_tokens_total:.2f}",
                    "checkpoint_path": checkpoint_path,
                }

                writer.writerow(row)
                f.flush()

                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "adj_tokens": int(tracker.adjusted_seen_tokens_total),
                })

                # Stop if exposure limit is reached
                if tracker.adjusted_seen_tokens_total >= max_adjusted_token_exposure:
                    logger.info(
                        f"Stopping: reached adjusted token exposure limit "
                        f"{max_adjusted_token_exposure:,}"
                    )
                    final_path = write_final_checkpoint_row(
                        epoch=epoch,
                        global_step=global_step,
                        train_loss=last_train_loss,
                    )
                    logger.info(f"Final checkpoint saved to {final_path}")
                    return

                if max_steps is not None and global_step >= max_steps:
                    logger.info(f"Stopping: reached max_steps={max_steps}")
                    final_path = write_final_checkpoint_row(
                        epoch=epoch,
                        global_step=global_step,
                        train_loss=last_train_loss,
                    )
                    logger.info(f"Final checkpoint saved to {final_path}")
                    return

        if last_train_loss is not None:
            final_path = write_final_checkpoint_row(
                epoch=epoch,
                global_step=global_step,
                train_loss=last_train_loss,
            )
        else:
            final_path = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                checkpoint_name="final_model",
            )

    logger.info("Training finished.")
    logger.info(f"Final checkpoint saved to {final_path}")
    logger.info(f"Training log saved to {log_path}")

def train_model_curriculum(
    model,
    stage_loaders,
    tokenizer,
    config,
):
    logger = logging.getLogger(__name__)

    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "training_log.csv")

    learning_rate = float(config.get("learning_rate", 5e-4))
    max_epochs = int(config.get("max_epochs", 1))

    max_steps = config.get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None

    max_val_steps = int(config.get("max_val_steps", 20))
    eval_every_steps = int(config.get("eval_every_steps", 10))
    gradient_accumulation_steps = int(
        config.get("gradient_accumulation_steps", 1)
    )
    target_effective_tokens_per_update = config.get(
        "target_effective_tokens_per_update", None
    )
    target_effective_tokens_per_update = (
        int(target_effective_tokens_per_update)
        if target_effective_tokens_per_update is not None
        else None
    )
    effective_tokens_per_update = (
        int(config["batch_size"])
        * int(config["max_length"])
        * gradient_accumulation_steps
    )

    max_adjusted_token_exposure = float(
        config.get("max_adjusted_token_exposure", 1_000_000_000)
    )
    checkpoint_interval_tokens = config.get("checkpoint_interval_tokens", None)
    checkpoint_interval_tokens = (
        int(checkpoint_interval_tokens)
        if checkpoint_interval_tokens is not None
        else None
    )
    checkpoint_schedule = config.get("checkpoint_schedule", "official")
    checkpoint_interval_for_schedule = (
        None if checkpoint_schedule == "official" else checkpoint_interval_tokens
    )
    save_intermediate_checkpoints = bool(
        config.get("save_intermediate_checkpoints", True)
    )

    if max_epochs > 10:
        raise ValueError("BabyLM rule: max_epochs must be <= 10")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if (
        target_effective_tokens_per_update is not None
        and effective_tokens_per_update != target_effective_tokens_per_update
    ):
        raise ValueError(
            "Effective tokens per optimizer update mismatch: "
            f"batch_size({config['batch_size']}) * "
            f"max_length({config['max_length']}) * "
            f"gradient_accumulation_steps({gradient_accumulation_steps}) = "
            f"{effective_tokens_per_update}, expected "
            f"{target_effective_tokens_per_update}."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info(f"Using device: {device}")
    logger.info(
        f"Gradient accumulation steps: {gradient_accumulation_steps} | "
        f"Effective tokens per optimizer update: {effective_tokens_per_update}"
    )
    logger.info(
        f"Save intermediate checkpoints: {save_intermediate_checkpoints}"
    )

    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    tracker = TokenExposureTracker()
    next_checkpoint_target = None
    if save_intermediate_checkpoints:
        next_checkpoint_target = get_next_checkpoint_target(
            tracker.adjusted_seen_tokens_total,
            checkpoint_interval_for_schedule,
        )
    global_step = 0
    micro_step = 0
    accumulated_micro_steps = 0
    accumulated_loss = 0.0
    optimizer.zero_grad()
    last_epoch = None
    last_stage_name = None
    last_stage_step = None
    last_micro_train_loss = None
    last_update_train_loss_avg = ""
    last_target_adjusted_tokens = None
    last_val_loader = None

    fieldnames = [
        "epoch",
        "stage",
        "global_step", #cumulative optimizer steps across all stages
        "stage_step", # number of optimizer steps within the current stage
        "micro_step",
        "gradient_accumulation_steps",
        "effective_tokens_per_update",
        "train_loss",
        "micro_train_loss",
        "update_train_loss_avg",
        "validation_loss", # cross entropy
        "perplexity",
        "train_validation_gap",# validation_loss - update_train_loss_avg
        "raw_seen_tokens_total",
        "raw_seen_tokens_eng",
        "raw_seen_tokens_nld",
        "raw_seen_tokens_zho",
        "adjusted_seen_tokens_eng",
        "adjusted_seen_tokens_nld",
        "adjusted_seen_tokens_zho",
        "adjusted_seen_tokens_total",
        "stage_target_adjusted_tokens",
        "checkpoint_path",
    ]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def write_curriculum_checkpoint_row(
            epoch,
            stage_name,
            stage_step,
            micro_train_loss,
            update_train_loss_avg,
            target_adjusted_tokens,
            val_loader,
            checkpoint_name,
        ):
            validation_loss_value = (
                evaluate(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    max_val_steps=max_val_steps,
                )
                if val_loader is not None
                else None
            )
            validation_loss = (
                f"{validation_loss_value:.6f}"
                if validation_loss_value is not None
                else ""
            )
            perplexity = ""
            train_validation_gap = ""
            if validation_loss_value is not None:
                perplexity = (
                    f"{math.exp(validation_loss_value):.6f}"
                    if validation_loss_value < 100
                    else "inf"
                )
                if update_train_loss_avg != "":
                    train_validation_gap = (
                        f"{validation_loss_value - update_train_loss_avg:.6f}"
                    )

            checkpoint_path = ""
            if checkpoint_name == "final_model":
                checkpoint_path = save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    checkpoint_name=checkpoint_name,
                )
            formatted_update_train_loss_avg = (
                f"{update_train_loss_avg:.6f}"
                if update_train_loss_avg != ""
                else ""
            )
            writer.writerow({
                "epoch": epoch,
                "stage": stage_name,
                "global_step": global_step,
                "stage_step": stage_step,
                "micro_step": micro_step,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_tokens_per_update": effective_tokens_per_update,
                "train_loss": formatted_update_train_loss_avg,
                "micro_train_loss": f"{micro_train_loss:.6f}",
                "update_train_loss_avg": formatted_update_train_loss_avg,
                "validation_loss": validation_loss,
                "perplexity": perplexity,
                "train_validation_gap": train_validation_gap,
                "raw_seen_tokens_total": tracker.raw_seen_tokens_total,
                "raw_seen_tokens_eng": tracker.raw_seen_tokens_eng,
                "raw_seen_tokens_nld": tracker.raw_seen_tokens_nld,
                "raw_seen_tokens_zho": tracker.raw_seen_tokens_zho,
                "adjusted_seen_tokens_eng": f"{tracker.adjusted_seen_tokens_eng:.2f}",
                "adjusted_seen_tokens_nld": f"{tracker.adjusted_seen_tokens_nld:.2f}",
                "adjusted_seen_tokens_zho": f"{tracker.adjusted_seen_tokens_zho:.2f}",
                "adjusted_seen_tokens_total": f"{tracker.adjusted_seen_tokens_total:.2f}",
                "stage_target_adjusted_tokens": (
                    f"{target_adjusted_tokens:.0f}"
                    if target_adjusted_tokens is not None
                    else ""
                ),
                "checkpoint_path": checkpoint_path,
            })
            f.flush()
            return checkpoint_path

        def save_curriculum_milestone_checkpoint(checkpoint_target):
            checkpoint_name = format_checkpoint_name(checkpoint_target)
            checkpoint_path = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                checkpoint_name=checkpoint_name,
            )
            logger.info(
                f"Saved curriculum checkpoint at adjusted token target "
                f"{checkpoint_target:,}: {checkpoint_path}"
            )
            return checkpoint_path

        def write_curriculum_milestone_checkpoint_row(
            checkpoint_target,
            epoch,
            stage_name,
            stage_step,
            micro_train_loss,
            update_train_loss_avg,
            target_adjusted_tokens,
        ):
            checkpoint_path = save_curriculum_milestone_checkpoint(
                checkpoint_target
            )
            formatted_update_train_loss_avg = (
                f"{update_train_loss_avg:.6f}"
                if update_train_loss_avg != ""
                else ""
            )
            writer.writerow({
                "epoch": epoch,
                "stage": stage_name,
                "global_step": global_step,
                "stage_step": stage_step,
                "micro_step": micro_step,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_tokens_per_update": effective_tokens_per_update,
                "train_loss": formatted_update_train_loss_avg,
                "micro_train_loss": (
                    f"{micro_train_loss:.6f}"
                    if micro_train_loss is not None
                    else ""
                ),
                "update_train_loss_avg": formatted_update_train_loss_avg,
                "validation_loss": "",
                "perplexity": "",
                "train_validation_gap": "",
                "raw_seen_tokens_total": tracker.raw_seen_tokens_total,
                "raw_seen_tokens_eng": tracker.raw_seen_tokens_eng,
                "raw_seen_tokens_nld": tracker.raw_seen_tokens_nld,
                "raw_seen_tokens_zho": tracker.raw_seen_tokens_zho,
                "adjusted_seen_tokens_eng": f"{tracker.adjusted_seen_tokens_eng:.2f}",
                "adjusted_seen_tokens_nld": f"{tracker.adjusted_seen_tokens_nld:.2f}",
                "adjusted_seen_tokens_zho": f"{tracker.adjusted_seen_tokens_zho:.2f}",
                "adjusted_seen_tokens_total": f"{tracker.adjusted_seen_tokens_total:.2f}",
                "stage_target_adjusted_tokens": (
                    f"{target_adjusted_tokens:.0f}"
                    if target_adjusted_tokens is not None
                    else ""
                ),
                "checkpoint_path": checkpoint_path,
            })
            f.flush()
            return checkpoint_path

        completed_stages = 0
        last_stage_checkpoint_path = ""

        for stage_index, stage in enumerate(stage_loaders):
            stage_name = stage["name"]
            train_loader = stage["train_loader"]
            val_loader = stage["val_loader"]
            is_last_stage = stage_index == len(stage_loaders) - 1
            target_adjusted_tokens = stage.get("target_adjusted_tokens", None)
            target_adjusted_tokens = (
                float(target_adjusted_tokens)
                if target_adjusted_tokens is not None
                else None
            )
            stage_step = 0
            stage_finished = False

            logger.info(
                f"Starting stage: {stage_name} "
                f"(target_adjusted_tokens={target_adjusted_tokens})"
            )

            for epoch in range(1, max_epochs + 1):
                logger.info(f"Starting epoch {epoch}/{max_epochs} | stage: {stage_name}")

                progress = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch} | {stage_name}",
                )

                for batch in progress:
                    languages = batch.pop("language", None)
                    language_ids = batch.pop("language_ids", None)

                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                    }

                    if language_ids is not None:
                        language_ids = language_ids.to(device)

                    batch_adjusted_delta = compute_adjusted_token_delta(
                        attention_mask=batch["attention_mask"],
                        languages=languages,
                        language_ids=language_ids,
                    )
                    current_adjusted_total = tracker.adjusted_seen_tokens_total
                    if save_intermediate_checkpoints:
                        while (
                            next_checkpoint_target is not None
                            and current_adjusted_total + batch_adjusted_delta
                            > next_checkpoint_target
                        ):
                            write_curriculum_milestone_checkpoint_row(
                                checkpoint_target=next_checkpoint_target,
                                epoch=last_epoch if last_epoch is not None else epoch,
                                stage_name=(
                                    last_stage_name
                                    if last_stage_name is not None
                                    else stage_name
                                ),
                                stage_step=last_stage_step,
                                micro_train_loss=last_micro_train_loss,
                                update_train_loss_avg=last_update_train_loss_avg,
                                target_adjusted_tokens=last_target_adjusted_tokens,
                            )
                            next_checkpoint_target = get_next_checkpoint_target(
                                next_checkpoint_target,
                                checkpoint_interval_for_schedule,
                            )

                    would_exceed_stage = (
                        target_adjusted_tokens is not None
                        and current_adjusted_total + batch_adjusted_delta
                        > target_adjusted_tokens
                    )
                    would_exceed_global = (
                        current_adjusted_total + batch_adjusted_delta
                        > max_adjusted_token_exposure
                    )

                    if would_exceed_stage or would_exceed_global:
                        optimizer.zero_grad()
                        accumulated_loss = 0.0
                        accumulated_micro_steps = 0

                        limit = (
                            max_adjusted_token_exposure
                            if would_exceed_global
                            else target_adjusted_tokens
                        )
                        logger.info(
                            "Stopping before next batch: adjusted token "
                            f"exposure would exceed {limit:,.0f} "
                            f"(current={current_adjusted_total:.2f}, "
                            f"batch_delta={batch_adjusted_delta:.2f})"
                        )

                        checkpoint_name = "final_model" if (
                            would_exceed_global or is_last_stage
                        ) else ""

                        if last_micro_train_loss is not None:
                            checkpoint_path = write_curriculum_checkpoint_row(
                                epoch=last_epoch,
                                stage_name=last_stage_name,
                                stage_step=last_stage_step,
                                micro_train_loss=last_micro_train_loss,
                                update_train_loss_avg=last_update_train_loss_avg,
                                target_adjusted_tokens=last_target_adjusted_tokens,
                                val_loader=last_val_loader,
                                checkpoint_name=checkpoint_name,
                            )
                        else:
                            checkpoint_path = ""
                            if checkpoint_name:
                                checkpoint_path = save_checkpoint(
                                    model=model,
                                    tokenizer=tokenizer,
                                    output_dir=output_dir,
                                    checkpoint_name=checkpoint_name,
                                )

                        stage_finished = True
                        if checkpoint_path:
                            last_stage_checkpoint_path = checkpoint_path

                        if would_exceed_global:
                            logger.info(
                                f"Final checkpoint saved to {checkpoint_path}"
                            )
                            logger.info(f"Training log saved to {log_path}")
                            return

                        logger.info(
                            f"Finished stage {stage_name} before exceeding "
                            f"target {target_adjusted_tokens:.0f}; "
                            f"current adjusted tokens "
                            f"{current_adjusted_total:.2f}"
                        )
                        break

                    outputs = model(**batch)
                    loss = outputs.loss

                    micro_train_loss = loss.item()
                    accumulated_loss += micro_train_loss
                    update_train_loss_avg = ""

                    scaled_loss = loss / gradient_accumulation_steps
                    scaled_loss.backward()

                    micro_step += 1
                    accumulated_micro_steps += 1

                    if language_ids is not None:
                        tracker.update_from_language_ids(
                            attention_mask=batch["attention_mask"],
                            language_ids=language_ids,
                        )
                    else:
                        tracker.update(
                            attention_mask=batch["attention_mask"],
                            languages=languages,
                        )

                    adjusted_total = tracker.adjusted_seen_tokens_total
                    checkpoint_path = ""
                    optimizer_updated = (
                        accumulated_micro_steps == gradient_accumulation_steps
                    )

                    if optimizer_updated:
                        optimizer.step()
                        optimizer.zero_grad()
                        update_train_loss_avg = (
                            accumulated_loss / gradient_accumulation_steps
                        )
                        accumulated_loss = 0.0
                        global_step += 1
                        stage_step += 1
                        accumulated_micro_steps = 0

                    if save_intermediate_checkpoints:
                        while (
                            next_checkpoint_target is not None
                            and adjusted_total >= next_checkpoint_target
                        ):
                            write_curriculum_milestone_checkpoint_row(
                                checkpoint_target=next_checkpoint_target,
                                epoch=epoch,
                                stage_name=stage_name,
                                stage_step=stage_step,
                                micro_train_loss=micro_train_loss,
                                update_train_loss_avg=update_train_loss_avg,
                                target_adjusted_tokens=target_adjusted_tokens,
                            )
                            next_checkpoint_target = get_next_checkpoint_target(
                                next_checkpoint_target,
                                checkpoint_interval_for_schedule,
                            )

                    validation_loss = ""
                    perplexity = ""
                    train_validation_gap = ""
                    if (
                        optimizer_updated
                        and val_loader is not None
                        and global_step % eval_every_steps == 0
                    ):
                        validation_loss_value = evaluate(
                            model=model,
                            val_loader=val_loader,
                            device=device,
                            max_val_steps=max_val_steps,
                        )
                        validation_loss = (
                            f"{validation_loss_value:.6f}"
                            if validation_loss_value is not None
                            else ""
                        )
                        if validation_loss_value is not None:
                            perplexity = (
                                f"{math.exp(validation_loss_value):.6f}"
                                if validation_loss_value < 100
                                else "inf"
                            )
                            if update_train_loss_avg != "":
                                train_validation_gap = (
                                    f"{validation_loss_value - update_train_loss_avg:.6f}"
                                )

                    formatted_update_train_loss_avg = (
                        f"{update_train_loss_avg:.6f}"
                        if update_train_loss_avg != ""
                        else ""
                    )
                    last_epoch = epoch
                    last_stage_name = stage_name
                    last_stage_step = stage_step
                    last_micro_train_loss = micro_train_loss
                    last_update_train_loss_avg = update_train_loss_avg
                    last_target_adjusted_tokens = target_adjusted_tokens
                    last_val_loader = val_loader
                    row = {
                        "epoch": epoch,
                        "stage": stage_name,
                        "global_step": global_step,
                        "stage_step": stage_step,
                        "micro_step": micro_step,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "effective_tokens_per_update": effective_tokens_per_update,
                        "train_loss": formatted_update_train_loss_avg,
                        "micro_train_loss": f"{micro_train_loss:.6f}",
                        "update_train_loss_avg": formatted_update_train_loss_avg,
                        "validation_loss": validation_loss,
                        "perplexity": perplexity,
                        "train_validation_gap": train_validation_gap,
                        "raw_seen_tokens_total": tracker.raw_seen_tokens_total,
                        "raw_seen_tokens_eng": tracker.raw_seen_tokens_eng,
                        "raw_seen_tokens_nld": tracker.raw_seen_tokens_nld,
                        "raw_seen_tokens_zho": tracker.raw_seen_tokens_zho,
                        "adjusted_seen_tokens_eng": f"{tracker.adjusted_seen_tokens_eng:.2f}",
                        "adjusted_seen_tokens_nld": f"{tracker.adjusted_seen_tokens_nld:.2f}",
                        "adjusted_seen_tokens_zho": f"{tracker.adjusted_seen_tokens_zho:.2f}",
                        "adjusted_seen_tokens_total": f"{tracker.adjusted_seen_tokens_total:.2f}",
                        "stage_target_adjusted_tokens": (
                            f"{target_adjusted_tokens:.0f}"
                            if target_adjusted_tokens is not None
                            else ""
                        ),
                        "checkpoint_path": checkpoint_path,
                    }

                    writer.writerow(row)
                    f.flush()

                    progress.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "adj_tokens": int(tracker.adjusted_seen_tokens_total),
                    })

                    if (
                        optimizer_updated
                        and
                        target_adjusted_tokens is not None
                        and adjusted_total >= target_adjusted_tokens
                    ):
                        checkpoint_name = "final_model" if is_last_stage else ""
                        checkpoint_path = write_curriculum_checkpoint_row(
                            epoch=epoch,
                            stage_name=stage_name,
                            stage_step=stage_step,
                            micro_train_loss=micro_train_loss,
                            update_train_loss_avg=update_train_loss_avg,
                            target_adjusted_tokens=target_adjusted_tokens,
                            val_loader=val_loader,
                            checkpoint_name=checkpoint_name,
                        )

                        if checkpoint_path:
                            logger.info(
                                f"Finished stage {stage_name} at adjusted tokens "
                                f"{adjusted_total:.2f}; saved {checkpoint_path}"
                            )
                        else:
                            logger.info(
                                f"Finished stage {stage_name} at adjusted tokens "
                                f"{adjusted_total:.2f}; intermediate checkpoint skipped"
                            )

                        stage_finished = True
                        if checkpoint_path:
                            last_stage_checkpoint_path = checkpoint_path
                        break

                    if (
                        optimizer_updated
                        and tracker.adjusted_seen_tokens_total
                        >= max_adjusted_token_exposure
                    ):
                        logger.info(
                            f"Stopping: reached adjusted token exposure limit "
                            f"{max_adjusted_token_exposure:,}"
                        )
                        final_path = write_curriculum_checkpoint_row(
                            epoch=epoch,
                            stage_name=stage_name,
                            stage_step=stage_step,
                            micro_train_loss=micro_train_loss,
                            update_train_loss_avg=update_train_loss_avg,
                            target_adjusted_tokens=target_adjusted_tokens,
                            val_loader=val_loader,
                            checkpoint_name="final_model",
                        )
                        logger.info(f"Final checkpoint saved to {final_path}")
                        return

                    if (
                        optimizer_updated
                        and max_steps is not None
                        and global_step >= max_steps
                    ):
                        logger.info(f"Stopping: reached max_steps={max_steps}")
                        final_path = write_curriculum_checkpoint_row(
                            epoch=epoch,
                            stage_name=stage_name,
                            stage_step=stage_step,
                            micro_train_loss=micro_train_loss,
                            update_train_loss_avg=update_train_loss_avg,
                            target_adjusted_tokens=target_adjusted_tokens,
                            val_loader=val_loader,
                            checkpoint_name="final_model",
                        )
                        logger.info(f"Final checkpoint saved to {final_path}")
                        return

                if stage_finished:
                    if accumulated_micro_steps != 0:
                        raise RuntimeError(
                            "Stage finished with incomplete gradient accumulation."
                        )
                    break

            if not stage_finished:
                logger.warning(
                    f"Stage {stage_name} ended without reaching target "
                    f"{target_adjusted_tokens}."
                )
            else:
                completed_stages += 1

        if completed_stages == len(stage_loaders) and last_stage_checkpoint_path:
            logger.info("Curriculum training finished.")
            logger.info(f"Final checkpoint saved to {last_stage_checkpoint_path}")
            logger.info(f"Training log saved to {log_path}")
            return

        if last_micro_train_loss is not None:
            final_path = write_curriculum_checkpoint_row(
                epoch=last_epoch,
                stage_name=last_stage_name,
                stage_step=last_stage_step,
                micro_train_loss=last_micro_train_loss,
                update_train_loss_avg=last_update_train_loss_avg,
                target_adjusted_tokens=last_target_adjusted_tokens,
                val_loader=last_val_loader,
                checkpoint_name="final_model",
            )
        else:
            final_path = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                checkpoint_name="final_model",
            )

        logger.info("Curriculum training finished.")
        logger.info(f"Final checkpoint saved to {final_path}")
        logger.info(f"Training log saved to {log_path}")
