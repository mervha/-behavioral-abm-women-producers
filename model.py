"""
Behavioral Agent-Based Model for Micro-scale Women Producers
============================================================

Author: [BLINDED for peer review]

Implements:
- 30 heterogeneous women producer agents (3 experience classes)
- 30 supplier agents with discrimination coefficients
- Watts-Strogatz small-world producer network
- Three endogenous mechanisms: advice flow, cooperative info-sharing,
  social contagion
- Adaptive supplier learning
- Endogenous exit via fatigue-threshold hazard
"""

import json
import numpy as np
import networkx as nx
import mesa


class ProducerAgent(mesa.Agent):
    """Micro-scale woman producer."""

    def __init__(self, model, experience_class):
        super().__init__(model)
        self.experience_class = experience_class  # 'new', 'mid', 'experienced'
        mult = model.params["experience_multipliers"][experience_class]
        self.negotiation_mult = mult["negotiation"]
        self.advice_mult = mult["advice"]
        self.fatigue_stock = 0.0
        self.cumulative_welfare = 0.0
        self.active = True
        self.exit_month = None
        self.cooperative_member = False

    def step(self):
        if not self.active:
            return

        # --- Supply-side burden ---
        supplier = self.random.choice(self.model.suppliers)
        delta = supplier.discrimination_coef

        price_asym = self.model.params["supply_side"]["omega_alpha"] * delta
        negotiation_loss = (self.model.params["supply_side"]["lambda_m"]
                            * delta * self.negotiation_mult)
        identity_tax = self.model.params["supply_side"]["c_t"]
        secondary_loss = self.model.params["supply_side"]["gamma"]
        info_cost = self.model.params["supply_side"]["c_L"]

        # Cooperative info-sharing reduces info cost
        if self.cooperative_member and self.model.policies.get("cooperative", False):
            for neighbor in self.model.network.neighbors(self.unique_id):
                neighbor_agent = self.model.producer_map.get(neighbor)
                if neighbor_agent and neighbor_agent.cooperative_member:
                    info_cost *= 0.5
                    break

        supply_burden = (price_asym + negotiation_loss + identity_tax
                         + secondary_loss + info_cost)

        # --- Demand-side advantage ---
        U_base = self.model.params["demand_side"]["U_base"]
        psi_w = self.model.params["demand_side"]["psi_w"]
        psi_p = self.model.params["demand_side"]["psi_p"]
        psi_k = self.model.params["demand_side"]["psi_k"]

        # Producer Signal policy boosts psi_w
        if self.model.policies.get("producer_signal", False):
            psi_w = 0.25

        # Community premium from advice flow (endogenous)
        active_neighbors = [n for n in self.model.network.neighbors(self.unique_id)
                            if self.model.producer_map.get(n)
                            and self.model.producer_map[n].active]
        if active_neighbors:
            avg_neighbor_welfare = np.mean([
                self.model.producer_map[n].cumulative_welfare
                for n in active_neighbors
            ])
            community_premium = psi_k * self.advice_mult * (avg_neighbor_welfare / 100.0)
        else:
            community_premium = 0.0

        demand_utility = U_base + psi_w + psi_p + community_premium

        # --- Period welfare ---
        period_welfare = demand_utility - supply_burden
        self.cumulative_welfare += period_welfare

        # --- Fatigue update ---
        self.fatigue_stock = max(
            0.0,
            self.fatigue_stock + max(0.0, supply_burden - demand_utility)
            - self.model.params["supply_side"]["rho"]
        )

        # --- Endogenous exit decision ---
        if self.model.exit_mode == "endogenous":
            theta = self.model.params["exit_dynamics"]["theta_S"]
            h0 = self.model.params["exit_dynamics"]["h_0"]
            h1 = self.model.params["exit_dynamics"]["h_1"]
            h2 = self.model.params["exit_dynamics"]["h_2"]

            hazard = h0
            if self.fatigue_stock > theta:
                hazard += h1 * (self.fatigue_stock - theta)
            # Social contagion: neighbors who recently exited
            recent_exits = sum(
                1 for n in self.model.network.neighbors(self.unique_id)
                if (self.model.producer_map.get(n)
                    and not self.model.producer_map[n].active
                    and self.model.producer_map[n].exit_month is not None
                    and (self.model.month - self.model.producer_map[n].exit_month) <= 12)
            )
            hazard += h2 * recent_exits

            if self.random.random() < hazard:
                self.active = False
                self.exit_month = self.model.month


class SupplierAgent(mesa.Agent):
    """Supplier with a discrimination coefficient that can adapt under training."""

    def __init__(self, model, is_female):
        super().__init__(model)
        self.is_female = is_female

        mu = (model.params["discrimination"]["mu_F"] if is_female
              else model.params["discrimination"]["mu_M"])
        sigma = model.params["discrimination"]["sigma"]
        self.discrimination_coef = max(0.0, self.random.gauss(mu, sigma))

    def step(self):
        # Adaptive learning under Supplier Training policy
        if self.model.policies.get("supplier_training", False):
            eta = self.model.params["supplier_learning"]["eta"]
            self.discrimination_coef *= (1 - eta)
            self.discrimination_coef = max(0.0, self.discrimination_coef)


class ProducerSupplierModel(mesa.Model):
    """Top-level ABM."""

    def __init__(self, params, seed, policies=None, exit_mode="endogenous"):
        super().__init__(seed=seed)
        self.params = params
        self.policies = policies or {}
        self.exit_mode = exit_mode
        self.month = 0

        # --- Producers ---
        n_p = params["agents"]["N_producers"]
        p_exp = params["agents"]["p_experienced"]
        p_mid = params["agents"]["p_mid"]

        producers = []
        for i in range(n_p):
            r = self.random.random()
            if r < p_exp:
                cls = "experienced"
            elif r < p_exp + p_mid:
                cls = "mid"
            else:
                cls = "new"
            producers.append(ProducerAgent(self, cls))

        # Cooperative membership under Cooperative policy
        if self.policies.get("cooperative", False):
            for p in producers:
                if self.random.random() < 0.6:
                    p.cooperative_member = True

        # --- Suppliers ---
        n_s = params["agents"]["N_suppliers"]
        p_f = params["agents"]["p_female_supplier"]
        # Supply Quota policy raises female supplier ratio
        if self.policies.get("supply_quota", False):
            p_f = 0.30

        suppliers = []
        for _ in range(n_s):
            is_f = self.random.random() < p_f
            suppliers.append(SupplierAgent(self, is_f))
        self.suppliers = suppliers

        # --- Watts-Strogatz network over producers ---
        k = params["network"]["k"]
        p_ws = params["network"]["rewiring_p"]
        self.network = nx.watts_strogatz_graph(n_p, k, p_ws, seed=seed)

        # Map producer.unique_id -> agent (Mesa assigns unique_id on creation)
        self.producer_map = {p.unique_id: p for p in producers}
        # Relabel network nodes (0..n-1) to actual unique_ids
        ids = list(self.producer_map.keys())
        mapping = {i: ids[i] for i in range(n_p)}
        self.network = nx.relabel_nodes(self.network, mapping)

    def step(self):
        self.month += 1
        # Producers act first (using suppliers' current coefficients)
        for p in list(self.producer_map.values()):
            p.step()
        # Suppliers update afterwards (adaptive learning)
        for s in self.suppliers:
            s.step()

    def run(self, T=None):
        T = T or self.params["simulation"]["T_months"]
        for _ in range(T):
            self.step()
        return self.collect_outcomes()

    def collect_outcomes(self):
        producers = list(self.producer_map.values())
        survival = np.mean([p.active for p in producers])
        welfare = np.mean([p.cumulative_welfare for p in producers])
        by_class = {}
        for cls in ("new", "mid", "experienced"):
            sub = [p for p in producers if p.experience_class == cls]
            if sub:
                by_class[cls] = {
                    "survival": np.mean([p.active for p in sub]),
                    "welfare": np.mean([p.cumulative_welfare for p in sub]),
                }
        return {
            "survival_rate": survival,
            "mean_welfare": welfare,
            "by_class": by_class,
        }


def load_params(path="params.json"):
    with open(path) as f:
        return json.load(f)


def load_seeds(path="seeds_used.txt"):
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                seeds.append(int(line))
    return seeds
