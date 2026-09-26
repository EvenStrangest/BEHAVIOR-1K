import omnigibson as og
from omnigibson.metrics.metric_base import MetricBase
from typing import Optional, Sequence


def compute_q_score(
    success: bool,
    now_satisfied_options: Sequence[Sequence[bool]],
    initial_satisfied_options: Sequence[Sequence[bool]],
) -> float:
    """
    Partial-success (Q-score) for one episode/env: a full success scores 1.0; otherwise the fraction
    of goal predicates that were NOT satisfied at episode start but ARE satisfied now, maximized over
    the alternative goal-state options. Mirrors the pre-refactor inline formula (lives here next to its
    only caller, TaskMetric). Empty options/no options return 0.0 instead of raising.
    """
    if success:
        return 1.0
    if not now_satisfied_options:
        return 0.0
    option_scores = []
    for now_opt, init_opt in zip(now_satisfied_options, initial_satisfied_options):
        if len(now_opt) == 0:
            option_scores.append(0.0)
            continue
        newly_satisfied = sum(int((not init) and now) for now, init in zip(now_opt, init_opt))
        option_scores.append(newly_satisfied / len(now_opt))
    return max(option_scores) if option_scores else 0.0


def _describe_predicate(pred):
    """Structured description of one ground goal predicate.

    The name lives in different places depending on the node: a BinaryAtomicFormula keeps its
    arguments in `body` and its name in the class attribute `STATE_NAME`, while a Negation
    keeps the whole raw form (name included) in `body` and the atom as its single child. So
    neither `str(body)` nor `STATE_NAME` alone identifies a predicate, and downstream analysis
    should not have to re-parse a rendered string -- hence the structured fields.
    """
    name = getattr(pred, "STATE_NAME", None)
    body = getattr(pred, "body", None)
    kids = list(getattr(pred, "children", None) or [])
    if name:
        args = [str(a).strip("?") for a in (body or [])]
        return {
            "predicate": "({} {})".format(name, " ".join(args)).strip(),
            "state": name,
            "negated": False,
            "args": args,
        }
    # Detect negation STRUCTURALLY (no STATE_NAME, exactly one child) rather than by matching
    # the class name. Ground goal-state options are conjunctions of literals -- De Morgan is
    # applied during grounding -- so a node with no predicate name and a single child is a
    # negated atom. Matching on the string "Negation" silently degrades to the unlabelled
    # fallback for any subclass or rename, and the degradation is invisible in the output.
    if len(kids) == 1:
        inner = _describe_predicate(kids[0])
        return {
            "predicate": "(not {})".format(inner["predicate"]),
            "state": inner["state"],
            "negated": not inner["negated"],
            "args": inner["args"],
        }
    return {
        "predicate": "({} {})".format(type(pred).__name__, body),
        "state": type(pred).__name__,
        "negated": False,
        "args": [],
    }


class TaskMetric(MetricBase):
    def __init__(self, human_stats: Optional[dict] = None, env_idx: int = 0, env_accessor=None):
        super().__init__(env_idx=env_idx, env_accessor=env_accessor)
        self.timesteps = 0
        self.human_stats = human_stats
        if human_stats is None:
            print("No human stats provided.")
        else:
            self.human_stats = {
                "steps": self.human_stats["length"],
            }

    def reset(self, env=None):
        env = self._resolve_env(env)
        self.state[self._scene(env)] = dict()
        self.timesteps = 0
        self.render_timestep = og.sim.get_rendering_dt()
        self.initial_predicate_states = (
            env.get_goal_option_satisfaction()
            if env is self.env_accessor
            else env.task.get_goal_option_satisfaction(self.env_idx)
        )

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        self.timesteps += 1
        return {"timesteps": self.timesteps}

    def _compute_episode_metrics(self, env, episode_info):
        # Use the accumulated state from episode_info
        timesteps = episode_info.get("timesteps", [])[-1] if episode_info.get("timesteps") else self.timesteps

        # task.success is a (num_envs,) bool tensor; read THIS env's slot. Partial credit (when not a
        # full success) counts newly-satisfied goal predicates per option, max over options.
        now_satisfied = (
            env.get_goal_option_satisfaction()
            if env is self.env_accessor
            else env.task.get_goal_option_satisfaction(self.env_idx)
        )
        final_q_score = compute_q_score(
            success=env.success if env is self.env_accessor else bool(env.task.success[self.env_idx]),
            now_satisfied_options=now_satisfied,
            initial_satisfied_options=self.initial_predicate_states,
        )

        # Per-predicate breakdown (terraforge; ported to the vectorized metric 2026-09-26 from the
        # 2026-08-27 single-env version). The aggregate q is not interpretable on its own: for
        # putting_shoes_on_rack 8 of 10 ground predicates are per-shoe `touching hallstand` / `not
        # touching floor`, so a shoe merely lifted off the floor scores credit. Derived from the SAME
        # satisfaction masks that compute_q_score consumes, so it can never disagree with `final`.
        # Masks are indexed by position within each ground_goal_state_options entry.
        task = env.shared_env.task if env is self.env_accessor else env.task
        options_detail = []
        for option, now_opt, init_opt in zip(task.ground_goal_state_options, now_satisfied, self.initial_predicate_states):
            preds = []
            for pred, now, init in zip(option, now_opt, init_opt):
                d = _describe_predicate(pred)
                d.update({"initially_true": bool(init), "final_true": bool(now), "newly_true": bool(now and not init)})
                preds.append(d)
            options_detail.append(preds)
        option_scores = [sum(d["newly_true"] for d in p) / len(p) if p else 0.0 for p in options_detail]
        best = max(range(len(option_scores)), key=option_scores.__getitem__) if option_scores else -1

        return {
            "q_score": {
                "final": final_q_score,
                "predicates": options_detail[best] if best >= 0 else [],
                "option_index": best,
                "n_options": len(options_detail),
            },
            "time": {
                "simulator_steps": timesteps,
                "simulator_time": timesteps * self.render_timestep,
                "normalized_time": self.human_stats["steps"] / timesteps if timesteps > 0 else float("inf"),
            },
        }
