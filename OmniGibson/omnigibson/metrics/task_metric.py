import omnigibson as og
from omnigibson.metrics.metric_base import MetricBase
from typing import Optional


class TaskMetric(MetricBase):
    def __init__(self, human_stats: Optional[dict] = None):
        super().__init__()
        self.timesteps = 0
        self.human_stats = human_stats
        if human_stats is None:
            print("No human stats provided.")
        else:
            self.human_stats = {
                "steps": self.human_stats["length"],
            }

    def reset(self, env):
        self.state[env.scene] = dict()
        self.timesteps = 0
        self.render_timestep = og.sim.get_rendering_dt()
        self.initial_predicate_states = [
            [pred.evaluate(env.task._evaluate_predicate) for pred in option]
            for option in env.task.ground_goal_state_options
        ]

    def _compute_step_metrics(self, env, action, obs, reward, terminated, truncated, info):
        self.timesteps += 1
        return {"timesteps": self.timesteps}

    def _compute_episode_metrics(self, env, episode_info):
        # Use the accumulated state from episode_info
        timesteps = episode_info.get("timesteps", [])[-1] if episode_info.get("timesteps") else self.timesteps

        # Evaluate every option ONCE and keep the per-predicate detail, then derive the score
        # from that same evaluation so the breakdown can never disagree with `final`.
        #
        # Added 2026-08-27 (terraforge). Upstream emits only the aggregate, but the aggregate
        # is not interpretable on its own: q counts predicates that became true AND were not
        # initially true, and for putting_shoes_on_rack 8 of the 10 ground predicates are
        # per-shoe `touching hallstand` / `not touching floor`. A shoe merely LIFTED off the
        # floor therefore scores credit without ever reaching the rack, so the same q can mean
        # "placed two shoes" or "picked up four and dropped them". The frozen pre-registration
        # (BEHAVIOR_1K_DESIGN_AND_PLAN.md section 4, predicate-type x channel-ablation matrix)
        # needs the split, and recovering it later would mean re-running every episode.
        # `final` is unchanged; the new keys are purely additive.
        options_detail = []
        for option, option_previous_state in zip(
            env.task.ground_goal_state_options, self.initial_predicate_states
        ):
            preds = []
            for pred, initially_true in zip(option, option_previous_state):
                now = bool(pred.evaluate(env.task._evaluate_predicate))
                preds.append(
                    {
                        "predicate": str(getattr(pred, "body", pred)),
                        "initially_true": bool(initially_true),
                        "final_true": now,
                        "newly_true": bool(now and not initially_true),
                    }
                )
            options_detail.append(preds)

        option_scores = [
            sum(d["newly_true"] for d in preds) / len(preds) for preds in options_detail if preds
        ]
        best = max(range(len(option_scores)), key=option_scores.__getitem__) if option_scores else -1

        if env.task.success:
            final_q_score = 1.0
        else:
            final_q_score = option_scores[best] if best >= 0 else 0.0

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
