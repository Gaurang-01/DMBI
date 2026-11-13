"""Plotly visualisations used in the dashboard."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORS = {"Legitimate": "#38bdf8", "Phishing": "#f87171"}


def class_distribution(data, target="CLASS_LABEL"):
    counts = data[target].map({0: "Legitimate", 1: "Phishing"}).value_counts().reset_index()
    counts.columns = ["Class", "Records"]
    return px.bar(counts, x="Class", y="Records", color="Class", color_discrete_map=COLORS, title="Website Class Distribution")


def correlation_heatmap(data, features):
    corr = data[features + ["CLASS_LABEL"]].corr(numeric_only=True)
    return px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature Correlation Heatmap", aspect="auto")


def feature_distribution(data, feature):
    plot = data.copy()
    plot["Class"] = plot["CLASS_LABEL"].map({0: "Legitimate", 1: "Phishing"})
    return px.histogram(plot, x=feature, color="Class", barmode="overlay", opacity=.7, color_discrete_map=COLORS, title=f"Distribution of {feature}")


def feature_importance(model, feature_names, top_n=15):
    importance = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).nlargest(top_n, "Importance").sort_values("Importance")
    return px.bar(importance, x="Importance", y="Feature", orientation="h", title=f"Top {top_n} Random Forest Feature Importances", color="Importance", color_continuous_scale="Blues")


def model_comparison(metrics):
    long = metrics.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], var_name="Metric", value_name="Score")
    return px.bar(long, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1.05], title="Model Performance Comparison")


def confusion_matrix_plot(matrix, name):
    return px.imshow(matrix, text_auto=True, x=["Predicted Legitimate", "Predicted Phishing"], y=["Actual Legitimate", "Actual Phishing"], color_continuous_scale="Blues", title=f"{name} Confusion Matrix")


def roc_plot(details):
    fig = go.Figure()
    for name, item in details.items():
        fig.add_trace(go.Scatter(x=item["fpr"], y=item["tpr"], mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random chance"))
    return fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
