"""Simple Streamlit dashboard for a Lunar Lander submission."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Lunar Lander Lab", page_icon="🚀", layout="wide")
st.title("🚀 Lunar Lander Lab")
st.caption("Train. Test on held-out seeds. Explain what changed.")

result_path = Path("evaluation.json")
if not result_path.exists():
    st.warning("Run `python evaluate.py --episodes 20` to create evaluation.json.")
    st.stop()

result = json.loads(result_path.read_text(encoding="utf-8"))
training = result.get("training", {})

columns = st.columns(6)
columns[0].metric("Algorithm", result.get("algorithm", "—").upper())
columns[1].metric("Mean reward", result.get("mean_reward", "—"))
columns[2].metric("Reward SD", result.get("reward_std", "—"))
columns[3].metric("Two-leg landing rate", f"{100 * result.get('two_leg_landing_rate', 0):.0f}%")
columns[4].metric("200-point rate", f"{100 * result.get('score_200_rate', 0):.0f}%")
columns[5].metric("Action-switch rate", result.get("mean_action_switch_rate", "—"))

st.subheader("Training budget")
st.write(
    f"{training.get('total_timesteps', '—'):,} timesteps · "
    f"{training.get('training_seconds', '—')} seconds · seed {training.get('seed', '—')}"
    if isinstance(training.get("total_timesteps"), int)
    else "Training summary unavailable"
)

episodes = pd.DataFrame(result.get("episodes", []))
if not episodes.empty:
    st.subheader("Official evaluation episodes")
    st.line_chart(episodes.set_index("seed")["reward"])
    st.dataframe(episodes, use_container_width=True, hide_index=True)

st.subheader("My improvement")
st.info("Replace this sentence: I changed ___ because ___; the evidence was ___.")

for video_name in ("landing.mp4", "landing.gif"):
    video_path = Path(video_name)
    if video_path.exists():
        st.subheader("Demonstration")
        if video_path.suffix == ".mp4":
            st.video(str(video_path))
        else:
            st.image(str(video_path))
        break
