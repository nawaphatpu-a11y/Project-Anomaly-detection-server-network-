import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from i18n import language_selector, t, type_label
from utils import ensure_evaluation_has_run

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data/processed"
MAIN_MODEL = "Isolation Forest (main)"
METRICS = ["precision", "recall", "f1"]
REAL_TYPES = ["CPU_SPIKE", "NETWORK_SURGE", "RESPONSE_DEGRADATION"]

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")
language_selector()
st.title("📈 Model Evaluation")
st.caption(t("eval.caption"))

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
                    x=[t("cm.pred_normal"), t("cm.pred_anomaly")], y=[t("cm.act_normal"), t("cm.act_anomaly")])
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=10))
    return fig


def alert_sentence(precision):
    real = round(precision * 100)
    return t("sentence.alerts", real=real, false=100 - real)


eval_df = read("evaluation_report.csv")
indep_before = read("evaluation_report_independent.csv")
retrain_holdout = read("evaluation_report_retrain_holdout.csv")
type_recall = read("recall_by_anomaly_type.csv")
cause_report = read("cause_diagnosis_report.csv")

if eval_df is None:
    st.warning(t("eval.not_found"))
    st.stop()

main = eval_df.loc[MAIN_MODEL]
others = eval_df.drop(index=MAIN_MODEL)
runner_up = t("delta.runner_up")

k1, k2, k3, k4 = st.columns(4)
k1.metric(t("metric.precision"), f"{main['precision']:.3f}", f"{main['precision'] - others['precision'].max():+.3f} {runner_up}")
k2.metric(t("metric.recall"), f"{main['recall']:.3f}", f"{main['recall'] - others['recall'].max():+.3f} {runner_up}")
k3.metric(t("metric.f1"), f"{main['f1']:.3f}", f"{main['f1'] - others['f1'].max():+.3f} {runner_up}")
k4.metric(t("metric.missed"), t("metric.missed_value", fn=f"{int(main['FN']):,}", total=f"{int(main['TP'] + main['FN']):,}"))
st.caption(t("eval.main_caption", model=MAIN_MODEL))

tab_models, tab_types, tab_indep = st.tabs([t("tab.models"), t("tab.types"), t("tab.independent")])

with tab_models:
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(metric_bars(eval_df, t("chart.metrics_by_model"), "model"), width="stretch")
    with right:
        chosen = st.selectbox(t("select.confusion"), eval_df.index.tolist(),
                              index=eval_df.index.tolist().index(MAIN_MODEL))
        st.plotly_chart(confusion_heatmap(eval_df.loc[chosen]), width="stretch")
        st.caption(alert_sentence(eval_df.loc[chosen, "precision"]))
    st.dataframe(eval_df.style.format({m: "{:.3f}" for m in METRICS}), width="stretch")
    best = eval_df["f1"].idxmax()
    st.success(t("success.best", model=best, f1=f"{eval_df.loc[best, 'f1']:.3f}"))

with tab_types:
    if type_recall is None:
        st.info(t("types.missing"))
    else:
        long = type_recall.reset_index().melt(id_vars="anomaly_type", var_name="model", value_name="recall")
        long["anomaly_type"] = long["anomaly_type"].map(type_label)
        fig = px.bar(long, x="anomaly_type", y="recall", color="model", barmode="group",
                     range_y=[0, 1], text_auto=".2f", title=t("types.chart_title"))
        fig.update_layout(xaxis_title="", legend_title_text="")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")
        weakest = type_recall[MAIN_MODEL].idxmin()
        text = t("types.weakest", model=MAIN_MODEL, type=type_label(weakest),
                 recall=f"{type_recall.loc[weakest, MAIN_MODEL]:.2f}")
        if weakest == "RESPONSE_DEGRADATION":
            text += t("types.weakest_reason")
        st.write(text)

    st.subheader(t("diag.header"))
    if cause_report is None:
        st.info(t("diag.missing"))
    else:
        real = cause_report.loc[REAL_TYPES]
        overall = (real[REAL_TYPES].values.diagonal().sum() / real["n"].sum()) if real["n"].sum() else 0
        c1, c2 = st.columns(2)
        c1.metric(t("diag.match"), f"{overall:.1%}")
        c2.metric(t("diag.false_unclassified"), f"{cause_report.loc['FALSE_ALARM', 'correct_rate']:.1%}")
        counts = cause_report.drop(columns=["n", "correct_rate"]).astype(int)
        fig = px.imshow(counts.values, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                        x=[t("diag.diagnosed", c=type_label("Other") if c == "UNCLASSIFIED" else type_label(c))
                           for c in counts.columns],
                        y=[t("diag.actual", i=t("cm.act_normal") if i == "FALSE_ALARM" else type_label(i))
                           for i in counts.index])
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(t("diag.caption"))

with tab_indep:
    if indep_before is None:
        st.info(t("indep.missing"))
    else:
        first = indep_before.iloc[0]
        rows = {t("indep.row_test"): main[METRICS], t("indep.row_full"): first[METRICS]}
        compare = pd.DataFrame(rows).T.astype(float)
        compare.index.name = "dataset"

        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(metric_bars(compare, t("indep.chart_title"), ""), width="stretch")
        with right:
            st.plotly_chart(confusion_heatmap(first), width="stretch")
            st.caption(alert_sentence(first["precision"]))

        st.write(t("indep.summary", recall=f"{first['recall']:.2f}", precision=f"{first['precision']:.2f}",
                   test_recall=f"{main['recall']:.2f}", test_precision=f"{main['precision']:.2f}"))
        st.dataframe(indep_before, width="stretch")
        scatter = PROCESSED / "evaluation_scatter_independent.png"
        if scatter.exists():
            st.image(str(scatter), caption=t("indep.scatter_caption"))

        if retrain_holdout is not None:
            st.subheader(t("indep.retrain_title"))
            sample = retrain_holdout.iloc[0][["TP", "FP", "TN", "FN"]].sum()
            st.caption(t("indep.retrain_note", n=f"{int(sample):,}"))
            st.plotly_chart(metric_bars(retrain_holdout, t("indep.retrain_title"), ""), width="stretch")
            st.dataframe(retrain_holdout, width="stretch")
