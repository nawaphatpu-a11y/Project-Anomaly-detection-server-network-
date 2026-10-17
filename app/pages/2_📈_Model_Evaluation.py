import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import ensure_evaluation_has_run

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data/processed"
MAIN_MODEL = "Isolation Forest (main)"
METRICS = ["precision", "recall", "f1"]

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")
st.title("📈 Model Evaluation")
st.caption("Offline evaluation against the ground-truth labels embedded in the data, not live data")

ensure_evaluation_has_run()


def read(name):
    path = PROCESSED / name
    return pd.read_csv(path, index_col=0) if path.exists() else None


def metric_bars(df, title, x_title):
    long = df.reset_index().melt(id_vars=df.index.name or "index", value_vars=METRICS,
                                 var_name="metric", value_name="score")
    fig = px.bar(long, x=long.columns[0], y="score", color="metric", barmode="group",
                 range_y=[0, 1], title=title, text_auto=".2f")
    fig.update_layout(xaxis_title=x_title, yaxis_title="score", legend_title_text="")
    fig.update_traces(textposition="outside")
    return fig


def confusion_heatmap(row):
    z = [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]
    fig = px.imshow(z, text_auto=",", color_continuous_scale="Blues", aspect="auto",
                    x=["Predicted normal", "Predicted anomaly"], y=["Actually normal", "Actually anomaly"])
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=10))
    return fig


def alert_sentence(precision):
    real = round(precision * 100)
    return f"Out of every 100 alerts, about {real} are real events and {100 - real} are false alarms"


eval_df = read("evaluation_report.csv")
indep_before = read("evaluation_report_independent.csv")
indep_after = read("evaluation_report_independent_after.csv")
type_recall = read("recall_by_anomaly_type.csv")

if eval_df is None:
    st.warning("data/processed/evaluation_report.csv not found. Run python evaluate.py first")
    st.stop()

main = eval_df.loc[MAIN_MODEL]
others = eval_df.drop(index=MAIN_MODEL)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Precision", f"{main['precision']:.3f}", f"{main['precision'] - others['precision'].max():+.3f} vs runner-up")
k2.metric("Recall", f"{main['recall']:.3f}", f"{main['recall'] - others['recall'].max():+.3f} vs runner-up")
k3.metric("F1", f"{main['f1']:.3f}", f"{main['f1'] - others['f1'].max():+.3f} vs runner-up")
k4.metric("Missed real events", f"{int(main['FN']):,} of {int(main['TP'] + main['FN']):,}")
st.caption(f"{MAIN_MODEL} on the test split, data the model never saw during training")

tab_models, tab_types, tab_indep = st.tabs(["Model Comparison", "By Anomaly Type", "Independent Test Set"])

with tab_models:
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(metric_bars(eval_df, "Precision / Recall / F1 by Model", "model"),
                        use_container_width=True)
    with right:
        chosen = st.selectbox("Confusion matrix for", eval_df.index.tolist(),
                              index=eval_df.index.tolist().index(MAIN_MODEL))
        st.plotly_chart(confusion_heatmap(eval_df.loc[chosen]), use_container_width=True)
        st.caption(alert_sentence(eval_df.loc[chosen, "precision"]))
    st.dataframe(eval_df.style.format({m: "{:.3f}" for m in METRICS}), use_container_width=True)
    best = eval_df["f1"].idxmax()
    st.success(f"Highest F1 on the test split: {best} ({eval_df.loc[best, 'f1']:.3f})")

with tab_types:
    if type_recall is None:
        st.info("recall_by_anomaly_type.csv not found. Run python evaluate.py again")
    else:
        long = type_recall.reset_index().melt(id_vars="anomaly_type", var_name="model", value_name="recall")
        fig = px.bar(long, x="anomaly_type", y="recall", color="model", barmode="group",
                     range_y=[0, 1], text_auto=".2f", title="Recall by Anomaly Type")
        fig.update_layout(xaxis_title="", legend_title_text="")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        weakest = type_recall[MAIN_MODEL].idxmin()
        text = f"{MAIN_MODEL} catches {weakest} least often ({type_recall.loc[weakest, MAIN_MODEL]:.2f})"
        if weakest == "RESPONSE_DEGRADATION":
            text += ", because the values rise gradually and the early part of each episode stays close to the normal range"
        st.write(text)

with tab_indep:
    if indep_before is None:
        st.info("evaluation_report_independent.csv not found")
    else:
        rows = {"Test split (same source as training)": main[METRICS]}
        rows["Independent set, before retrain"] = indep_before.iloc[0][METRICS]
        if indep_after is not None:
            rows["Independent set, after retrain"] = indep_after.iloc[0][METRICS]
        compare = pd.DataFrame(rows).T.astype(float)
        compare.index.name = "dataset"

        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(metric_bars(compare, "Same-source test split vs data from a different AI", ""),
                            use_container_width=True)
        with right:
            first = indep_before.iloc[0]
            st.plotly_chart(confusion_heatmap(first), use_container_width=True)
            st.caption(alert_sentence(first["precision"]))

        st.write(f"The model still catches almost every real event (recall {first['recall']:.2f}), "
                 f"but precision drops to {first['precision']:.2f} because normal values in this dataset differ from what the model saw in training. "
                 "Using it on another system would require recalibrating the threshold first")
        st.dataframe(pd.concat([indep_before, indep_after]) if indep_after is not None else indep_before,
                     use_container_width=True)
        scatter = PROCESSED / "evaluation_scatter_independent.png"
        if scatter.exists():
            st.image(str(scatter), caption="Caught, missed and falsely flagged points on the independent set")
