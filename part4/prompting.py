"""
Prompting utilities for multiple-choice QA.
fsample submission.
"""
import torch
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax

class PromptTemplate:
    TEMPLATES = {
        "basic": "{few_shot}Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\n{few_shot}Passage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{few_shot}{context}\n{question}\n{choices_formatted}\nThe answer is",
    }
    
    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, choice_format: str = "letter"):
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format
    
    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"] if self.choice_format == "letter" else [str(i+1) for i in range(len(choices))]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    
    def format(self, context: str, question: str, choices: List[str], few_shot='', **kwargs) -> str:
        return self.template.format(context=context, question=question, choices_formatted=self._format_choices(choices), few_shot=few_shot, **kwargs)
    
    def format_with_answer(self, context: str, question: str, choices: List[str], answer_idx: int) -> str:
        prompt = self.format(context, question, choices, few_shot='')
        label = chr(ord('A') + answer_idx) if self.choice_format == "letter" else str(answer_idx + 1)
        return f"{prompt} {label}"


class PromptingPipeline:
    def __init__(self, model, tokenizer, template: Optional[PromptTemplate] = None, device: str = "cuda", few_shot=None):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        self.few_shot = few_shot
        self._setup_choice_tokens()
    
    def _setup_choice_tokens(self):
        self.choice_tokens = {}
        for label in ["A", "B", "C", "D"]:
            for prefix in ["", " "]:
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    break

    def _create_few_shot(self):
        prefix = ""
        for fs in self.few_shot:
            correct = fs["choices"][fs["answer"]]
            prefix += f"{fs['context']}\nQuestion: {fs['question']}\nAnswer: {correct}\n\n"
        return prefix
    
    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        few_shot = self._create_few_shot() if self.few_shot else ""
        prompt = self.template.format(context, question, choices, few_shot)
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        logits = self.model(input_ids)[:, -1, :]
        
        choice_labels = ["A", "B", "C", "D"][:len(choices)]
        choice_logits = []
        for label in choice_labels:
            if label in self.choice_tokens:
                choice_logits.append(logits[0, self.choice_tokens[label]].item())
            else:
                choice_logits.append(float("-inf"))
        
        choice_logits = torch.tensor(choice_logits)
        probs = softmax(choice_logits, dim=-1)
        prediction = probs.argmax().item()
        
        if return_probs:
            return prediction, probs.tolist()
        return prediction
    
    @torch.no_grad()
    def predict_batch(self, fsamples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        return [self.predict_single(fs["context"], fs["question"], fs["choices"]) for fs in fsamples]


def evaluate_prompting(pipeline, fsamples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(fsamples, batch_size)
    correct = sum(1 for p, fs in zip(predictions, fsamples) if fs.get("answer", -1) >= 0 and p == fs["answer"])
    total = sum(1 for fs in fsamples if fs.get("answer", -1) >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}
