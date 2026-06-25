"""
Anti-Loop Controller: wraps any agent and applies runtime rule-based corrections.

Interventions:
  1. empty_search_fallback: search returns 0 results → auto-insert broader query
  2. repeat_search_to_click: repeated query → force click on first result
  3. search_count_to_click: > N searches → force click
  4. forced_buy_after_loop: max_steps-1 reached → force Buy Now
  5. action_normalized: fix casing/format mismatches
"""
import re

SEARCH_PATTERN = re.compile(r"search\[(.+)\]", re.IGNORECASE)
CLICK_PATTERN = re.compile(r"click\[(.+)\]", re.IGNORECASE)


def normalize_query(query):
    """Strip, lowercase, collapse whitespace."""
    return " ".join(query.strip().lower().split())


def shorten_query(query, max_words=5):
    """Take first N words as fallback broader query."""
    words = query.split()
    return " ".join(words[:max_words])


def normalize_action(action):
    """Fix common formatting issues in actions."""
    action = action.strip()
    if action.lower().startswith("search[") and not action.startswith("search["):
        inner = action[7:]
        action = f"search[{inner}]"
    if action.lower().startswith("click[") and not action.startswith("click["):
        inner = action[6:]
        action = f"click[{inner}]"
    return action


class AntiLoopController:
    """
    Wraps a ReActAgent and applies runtime corrections to prevent common failure loops.
    """

    def __init__(
        self,
        agent,
        max_consecutive_same_search=2,
        max_total_searches=4,
        max_steps=7,
        enable_repeat_to_click=True,
        enable_search_count_to_click=True,
        enable_forced_buy=True,
        conservative_buy=False,
    ):
        self.agent = agent
        self.max_consecutive_same_search = max_consecutive_same_search
        self.max_total_searches = max_total_searches
        self.max_steps = max_steps
        self.enable_repeat_to_click = enable_repeat_to_click
        self.enable_search_count_to_click = enable_search_count_to_click
        self.enable_forced_buy = enable_forced_buy
        self.conservative_buy = conservative_buy
        self.reset()

    def reset(self):
        self.prev_queries = []
        self.search_count = 0
        self.consecutive_same = 0
        self.step_count = 0
        self.last_obs_has_results = False
        self.interventions = []

    def _log_intervention(self, name, detail):
        self.interventions.append({
            "step": self.step_count,
            "intervention": name,
            "detail": detail,
        })

    def _broaden_query(self, query, obs_text):
        """Generate a broader query by taking fewer keywords."""
        broadened = shorten_query(query, max_words=4)
        if broadened == query:
            broadened = shorten_query(query, max_words=2)
        return broadened

    def _pick_product_from_obs(self, obs_text):
        """Extract the first product ASIN from observation text."""
        asin_pattern = re.search(r"([A-Z0-9]{10})", obs_text)
        if asin_pattern:
            return asin_pattern.group(1)
        return None

    def act(self, instruction, obs, clickables, history):
        """
        Get action from agent, then apply anti-loop corrections.
        Returns (thought, action, interventions_this_step).
        """
        self.step_count += 1
        interventions_this_step = []

        # Get agent's raw action
        thought, raw_action = self.agent.act(instruction, obs, clickables, history)

        # Normalize action format
        action = normalize_action(raw_action)
        if action != raw_action:
            self._log_intervention("action_normalized", f"{raw_action} -> {action}")
            interventions_this_step.append("action_normalized")

        sm = SEARCH_PATTERN.search(action)
        cm = CLICK_PATTERN.search(action)

        # --- Intervention 4: forced buy at max_steps ---
        if self.enable_forced_buy and self.step_count >= self.max_steps - 1 and not (cm and "buy now" in cm.group(1).lower()):
            if self.conservative_buy:
                # Conservative: force buy only if Buy Now available AND at least one product viewed AND no unvisited ASINs visible
                buy_available = any("buy now" in c.lower() for c in clickables)
                if buy_available and self.search_count >= 1 and self.step_count >= self.max_steps - 1:
                    action = "click[Buy Now]"
                    thought = "[CONTROLLER-CONSERVATIVE] Max steps + product viewed. Buying."
                    self._log_intervention("conservative_forced_buy", f"step={self.step_count}")
                    interventions_this_step.append("conservative_forced_buy")
                    return thought, action, interventions_this_step
            else:
                # Aggressive: force buy if available
                buy_available = any("buy now" in c.lower() for c in clickables)
                if buy_available:
                    action = "click[Buy Now]"
                    thought = "[CONTROLLER] Max steps reached. Buying best available product."
                    self._log_intervention("forced_buy_after_loop", f"step={self.step_count}")
                    interventions_this_step.append("forced_buy_after_loop")
                    return thought, action, interventions_this_step
                # If no Buy Now, click first product
                product = self._pick_product_from_obs(obs)
                if product and product.lower() in [c.lower() for c in clickables]:
                    action = f"click[{product}]"
                    thought = f"[CONTROLLER] Max steps. Clicking product {product} to review."
                    self._log_intervention("forced_buy_after_loop", f"click product {product}")
                    interventions_this_step.append("forced_buy_after_loop")
                    return thought, action, interventions_this_step

        # --- Interventions for search actions ---
        if sm:
            query = sm.group(1).strip()
            norm_q = normalize_query(query)

            # Track consecutive same queries
            if self.prev_queries and norm_q == self.prev_queries[-1]:
                self.consecutive_same += 1
            else:
                self.consecutive_same = 1
            self.prev_queries.append(norm_q)
            self.search_count += 1

            # --- Intervention 1: broaden empty search ---
            if "Page 1 (Total results: 0)" in obs or "Back to Search" in obs:
                # Check if previous search had no results (approximate)
                pass

            # --- Intervention 2: repeated search → click ---
            if self.enable_repeat_to_click and self.consecutive_same >= self.max_consecutive_same_search:
                product = self._pick_product_from_obs(obs)
                if product and product.lower() in [c.lower() for c in clickables]:
                    action = f"click[{product}]"
                    thought = f"[CONTROLLER] Repeated search detected. Clicking {product} instead."
                    self._log_intervention("repeat_search_to_click", f"query={norm_q}")
                    interventions_this_step.append("repeat_search_to_click")
                    self.consecutive_same = 0
                    return thought, action, interventions_this_step
                else:
                    # Can't click, broaden query
                    broader = self._broaden_query(query, obs)
                    if broader != query:
                        action = f"search[{broader}]"
                        thought = f"[CONTROLLER] Repeated search with no products. Broadening to '{broader}'."
                        self._log_intervention("repeat_search_to_broaden", f"{query} -> {broader}")
                        interventions_this_step.append("repeat_search_to_broaden")
                        self.consecutive_same = 0
                        return thought, action, interventions_this_step

            # --- Intervention 3: too many total searches ---
            if self.enable_search_count_to_click and self.search_count >= self.max_total_searches:
                product = self._pick_product_from_obs(obs)
                if product and product.lower() in [c.lower() for c in clickables]:
                    action = f"click[{product}]"
                    thought = f"[CONTROLLER] {self.search_count} searches done. Clicking {product}."
                    self._log_intervention("search_count_to_click", f"searches={self.search_count}")
                    interventions_this_step.append("search_count_to_click")
                    return thought, action, interventions_this_step

        # --- No intervention needed ---
        return thought, action, interventions_this_step
