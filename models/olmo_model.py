"""
models/olmo_loader.py

Utilities to load OLMo (and other HuggingFace causal-LM) checkpoints and
run batched text generation.

Models are loaded ONE AT A TIME by the caller to avoid OOM on a single GPU.
Use CheckpointModel.unload() to free GPU memory before loading the next model.
"""

import gc
import logging
from typing import Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GenerationConfig, GENERATION

logger = logging.getLogger(__name__)


class CheckpointModel:
    """
    Thin wrapper around a HuggingFace causal-LM model + tokenizer.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g. "allenai/OLMo-7B-SFT").
    checkpoint_name : str
        Human-readable tag used in result files (e.g. "sft__none").
    system_prompt : str | None
        Optional system prompt prepended to every user message.
    gen_config : GenerationConfig
        Generation hyper-parameters.
    """

    def __init__(
        self,
        model_id: str,
        checkpoint_name: str,
        system_prompt: Optional[str] = None,
        gen_config: GenerationConfig = GENERATION,
    ) -> None:
        self.model_id = model_id
        self.checkpoint_name = checkpoint_name
        self.system_prompt = system_prompt
        self.gen_config = gen_config

        logger.info("Loading tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        # Decoder-only models require left-padding for correct batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model: %s", model_id)
        self.model = self._load_model(model_id, gen_config)
        self.model.eval()
        logger.info("Model loaded: %s", model_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_id: str, gen_config: GenerationConfig):
        kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=gen_config.device, **kwargs
            )
            return model
        except (AttributeError, NotImplementedError) as e:
            logger.warning(
                "device_map='auto' failed (%s). Falling back to manual CUDA placement.", e
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return model.to(device)

    def _build_prompt(self, user_message: str) -> str:
        """Build the full prompt string with optional system prompt."""
        if self.tokenizer.chat_template:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_message})
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass  # fall through to manual format

        # Manual fallback (e.g. base model with no chat template)
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}\n")
        parts.append(f"User: {user_message}\nAssistant:")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for a list of raw user prompts.
        Returns a list of generated *response* strings (input stripped).
        """
        full_prompts = [self._build_prompt(p) for p in prompts]

        inputs = self.tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        # OLMo does not accept token_type_ids
        inputs.pop("token_type_ids", None)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.gen_config.max_new_tokens,
            temperature=self.gen_config.temperature if self.gen_config.do_sample else 1.0,
            do_sample=self.gen_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens
        responses = self.tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )
        return responses

    def generate(self, prompt: str) -> str:
        """Convenience wrapper for a single prompt."""
        return self.generate_batch([prompt])[0]

    def unload(self) -> None:
        """Delete model weights and free GPU memory."""
        logger.info("Unloading model: %s", self.checkpoint_name)
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __repr__(self) -> str:
        sp = "no system prompt" if self.system_prompt is None else "with system prompt"
        return f"CheckpointModel(name={self.checkpoint_name!r}, model={self.model_id!r}, {sp})"


def iter_checkpoints(
    checkpoint_map: dict,
    system_prompts: dict,
    gen_config: GenerationConfig = GENERATION,
) -> Iterator[CheckpointModel]:
    """
    Yield one CheckpointModel at a time (checkpoint × system_prompt).

    The caller MUST call model.unload() after processing each model
    to free GPU memory before the next one is loaded.

    Usage
    -----
        for model in iter_checkpoints(OLMO_CHECKPOINTS, SYSTEM_PROMPTS):
            responses = run_generation(model, prompts)
            model.unload()
    """
    for ckpt_name, model_id in checkpoint_map.items():
        for prompt_name, prompt_text in system_prompts.items():
            tag = f"{ckpt_name}__{prompt_name}"
            logger.info("Loading model variant: %s", tag)
            yield CheckpointModel(
                model_id=model_id,
                checkpoint_name=tag,
                system_prompt=prompt_text,
                gen_config=gen_config,
            )
