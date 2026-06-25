"""
ReAct Baseline Agent for WebShop.
Uses Qwen2.5-7B-Instruct (FP16 on 40GB GPU, optional 4-bit on 10GB GPU).
"""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Action patterns
SEARCH_PATTERN = re.compile(r"search\[(.+)\]", re.IGNORECASE)
CLICK_PATTERN = re.compile(r"click\[(.+)\]", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a web shopping agent. Your task is to find and buy a product "
    "matching the user's instruction. You can search for products, click on them "
    "to view details, and click 'Buy Now' to purchase.\n\n"
    "Available actions:\n"
    "- search[keywords]: Search for products using keywords\n"
    "- click[ASIN]: Click a product to view its details\n"
    "- click[Buy Now]: Purchase the current product\n"
    "- click[< Prev] / click[Next >]: Navigate search results\n"
    "- click[Back to Search]: Return to search results\n"
    "- click[Description] / click[Features] / click[Reviews]: View product details\n"
    "- click[option_value]: Select a product option (size, color, etc.)\n\n"
    "IMPORTANT RULES:\n"
    "1. Always start your response with a brief Thought, then output Action:\n"
    "2. Match ALL constraints from the instruction (color, size, price, style, etc.)\n"
    "3. Compare multiple products before buying if possible.\n"
    "4. Don't buy until you are confident the product matches ALL requirements.\n"
    "5. Format: Thought: <your reasoning>\nAction: search[keywords] or Action: click[element]\n"
)

FEW_SHOT_EXAMPLE = """
Instruction: Find me a slim fit men's dress shirt with French cuffs, size medium, in white or light blue, under $50.

Observation: [Search Result page]
[button] Back to Search [button_]
[SEP] Page 1 (Total results: 10)
[SEP] [button] B08KH2M822 [button_]
[SEP] Men's Slim Fit Dress Shirt, French Cuffs, White, Medium [SEP] $42.99
[SEP] [button] B08XYZ123 [button_]
[SEP] Men's Classic Fit Dress Shirt, Button Cuffs, Blue, Medium [SEP] $38.00
[SEP] [button] B09ABC456 [button_]
[SEP] Men's Slim Fit Dress Shirt, French Cuffs, Light Blue, Large [SEP] $45.50

Available actions: search, click[B08KH2M822], click[B08XYZ123], click[B09ABC456], click[Next >]

Thought: I need a slim fit, French cuffs, medium, white/light blue, under $50. B08KH2M822 matches all constraints: slim fit, French cuffs, white, medium, $42.99. Let me click to see full details.

Action: click[B08KH2M822]
"""


class ReActAgent:
    def __init__(self, model_path, use_4bit=False, system_prompt=None):
        self.system_prompt = system_prompt  # None = use default SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

    def _build_prompt(self, instruction, observation, available_actions, history):
        acts_str = ", ".join(available_actions[:30])
        system = self.system_prompt if self.system_prompt else SYSTEM_PROMPT
        parts = [system, FEW_SHOT_EXAMPLE]

        for h in history[-3:]:
            parts.append(f"Instruction: {h['instruction']}")
            parts.append(f"Observation: {h['observation'][:1500]}")
            parts.append(f"Thought: {h['thought']}")
            parts.append(f"Action: {h['action']}")

        parts.append(f"Instruction: {instruction}")
        parts.append(f"Observation: {observation[:2000]}")
        parts.append(f"Available actions: {acts_str}")
        parts.append("Thought:")

        prompt = "\n".join(parts)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return text

    def act(self, instruction, observation, available_actions, history=None):
        if history is None:
            history = []

        prompt = self._build_prompt(instruction, observation, available_actions, history)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        thought, action = self._parse_response(response)
        return thought, action

    def _parse_response(self, response):
        thought = ""
        action = "search[ ]"

        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        search_match = SEARCH_PATTERN.search(response)
        click_match = CLICK_PATTERN.search(response)

        if search_match:
            action = f"search[{search_match.group(1).strip()}]"
        elif click_match:
            action = f"click[{click_match.group(1).strip()}]"

        return thought, action

    def parse_action(self, action_str):
        """Parse an action string into (action_type, argument)."""
        search = SEARCH_PATTERN.search(action_str)
        if search:
            return "search", search.group(1).strip()
        click = CLICK_PATTERN.search(action_str)
        if click:
            return "click", click.group(1).strip()
        return None, None
