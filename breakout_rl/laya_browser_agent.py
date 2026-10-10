"""Fixed-question browser adapter for the pinned Laya model; research stays upstream."""

from copy import deepcopy

import numpy as np


class LayaBrowserAgent:
    def __init__(self, agent, questions):
        from PIL import Image
        from laya.vlm import (QTYPES, build_vlm_inputs, collate_vlm, truncation_answer,
                              truncation_error, vlm_prefix)

        self.agent = agent
        self.device = agent.device
        self.questions = deepcopy(questions)
        self.mode = "fixed-eager"
        state = {"image": Image.fromarray(np.zeros((210, 160, 3), dtype=np.uint8))}
        prefix = vlm_prefix(agent.processor, [state["image"]], agent.prep)
        question = agent._to_internal(questions["move"])
        max_len, head_max_len = agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256)
        item = build_vlm_inputs(agent.processor, state, question, max_len, head_max_len,
                                prefix=prefix)
        truncated = truncation_answer(item["truncation"], question)
        if truncated:
            raise truncation_error("move", truncated, max_len, head_max_len)
        if len(item["markers"]) != 4 or prefix["raw_images"] is None:
            raise ValueError("Browser adapter requires four choices and GPU preprocessing")
        item["qtype"] = QTYPES[question["t"]]
        batch = collate_vlm([item], agent.processor.tokenizer.pad_token_id, with_pixels=False)
        self.args = [batch[k].to(self.device) for k in
                     ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        self.span = batch["option_span"].to(self.device)
        self.pixels = prefix["raw_images"].to(self.device)
        self.temperature = agent._checkpoint_temperature(item["qtype"], 4)
        self.labels = tuple(questions["move"]["criteria"])

    def _forward(self):
        features = self.agent.model.encode_raw_images(self.pixels)
        return self.agent.model(*self.args, image_hidden_states=features,
                                option_span=self.span)[0]

    def predict(self, state, questions, *, strict=True):
        import torch

        if questions != self.questions or not strict or set(state) != {"image"}:
            raise ValueError("Browser adapter accepts only its fixed strict Breakout question")
        rgb = np.asarray(state["image"])
        if rgb.shape != (210, 160, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected one 210 x 160 uint8 RGB frame")
        with torch.no_grad():
            self.pixels.copy_(torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0))
            logits = self._forward()
            scores = logits.float().cpu().numpy()[0].astype(np.float64)
        scores /= max(1e-3, self.temperature)
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        from laya.vlm import confidence_from_probs
        return {"answers": {"move": {"type": "choice",
                "choice": self.labels[int(probabilities.argmax())],
                "confidence": round(confidence_from_probs(probabilities, 4), 4),
                "probabilities": {label: round(float(p), 4)
                                  for label, p in zip(self.labels, probabilities)}}}}
