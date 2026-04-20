import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from observability import get_all_metrics

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'support.db')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="QuickBite AI Support Analytics",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        border-left: 4px solid #2E75B6;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1F4E79;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #2E75B6;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=QuickBite", width=200)
    st.title("AI Support Analytics")
    st.markdown("---")
    st.markdown("**System Status**")
    st.success("All systems operational")
    st.markdown("---")

    if st.button("Refresh Metrics", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    Intelligent Customer Support Platform
    monitoring resolution quality, routing
    efficiency, and AI system performance.
    """)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_metrics():
    return get_all_metrics()

@st.cache_data(ttl=60)
def load_interaction_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    logs = conn.execute('''
        SELECT
            log_id,
            query_text,
            resolution_status as routed_to,
            llm_response,
            tokens_used_input,
            tokens_used_output,
            latency_ms,
            escalated_to_human,
            human_feedback_score,
            retrieval_scores,
            tools_called,
            created_at
        FROM interaction_logs
        ORDER BY created_at DESC
    ''').fetchall()
    conn.close()
    return [dict(log) for log in logs]

metrics = load_metrics()
logs = load_interaction_logs()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🍔 QuickBite AI Support Analytics")
st.markdown(f"*Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}*")
st.markdown("---")

# ─────────────────────────────────────────────
# PANEL 1 — SYSTEM OVERVIEW (North Star Metrics)
# ─────────────────────────────────────────────

st.markdown('<div class="section-header">System Overview</div>',
            unsafe_allow_html=True)

rr = metrics["resolution_rate"]
ld = metrics["latency_distribution"]
fs = metrics["feedback_summary"]
rq = metrics["retrieval_quality"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    resolution_pct = rr.get("resolution_rate_pct", 0) or 0
    color = "normal" if resolution_pct >= 80 else "inverse"
    st.metric(
        label="Resolution Rate",
        value=f"{resolution_pct}%",
        delta=f"{rr.get('auto_resolved', 0)} of {rr.get('total_interactions', 0)} resolved"
    )

with col2:
    p50 = ld.get("p50_ms", 0)
    st.metric(
        label="Median Latency (P50)",
        value=f"{p50:,}ms",
        delta=f"P99: {ld.get('p99_ms', 0):,}ms",
        delta_color="inverse"
    )

with col3:
    avg_score = fs.get("avg_score") or 0
    st.metric(
        label="Avg Feedback Score",
        value=f"{avg_score}/5" if avg_score else "No data",
        delta=f"{fs.get('positive', 0)} positive, {fs.get('negative', 0)} negative"
    )

with col4:
    avg_sim = rq.get("avg_top_similarity_score", 0)
    st.metric(
        label="Avg Retrieval Score",
        value=f"{avg_sim:.2f}",
        delta=f"{rq.get('high_confidence_retrievals', 0)} high confidence"
    )

st.markdown("---")

# ─────────────────────────────────────────────
# PANEL 2 — ROUTING DISTRIBUTION
# ─────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-header">Routing Distribution</div>',
                unsafe_allow_html=True)

    routing_data = metrics["routing_distribution"]
    if routing_data:
        df_routing = pd.DataFrame(routing_data)

        color_map = {
            "faq_lookup": "#2ECC71",
            "faq_no_match": "#F39C12",
            "rag_pipeline": "#2E75B6",
            "agent": "#E74C3C"
        }

        fig_routing = px.pie(
            df_routing,
            values="interaction_count",
            names="routed_to",
            color="routed_to",
            color_discrete_map=color_map,
            hole=0.4
        )
        fig_routing.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )
        fig_routing.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        st.plotly_chart(fig_routing, use_container_width=True)

        # Token cost per tier table
        st.markdown("**Avg tokens per routing tier:**")
        for row in routing_data:
            tier_color = color_map.get(row["routed_to"], "#999")
            st.markdown(
                f"- **{row['routed_to']}**: "
                f"{int(row['avg_tokens']):,} tokens avg | "
                f"{row['pct_of_total']}% of traffic"
            )
    else:
        st.info("No routing data available yet.")

# ─────────────────────────────────────────────
# PANEL 3 — COST ANALYSIS
# ─────────────────────────────────────────────

with col_right:
    st.markdown('<div class="section-header">Cost Analysis by Tier</div>',
                unsafe_allow_html=True)

    cost_data = metrics["cost_analysis"]
    if cost_data:
        df_cost = pd.DataFrame(cost_data)

        fig_cost = px.bar(
            df_cost,
            x="routed_to",
            y="estimated_cost_usd",
            color="routed_to",
            color_discrete_map={
                "faq_lookup": "#2ECC71",
                "faq_no_match": "#F39C12",
                "rag_pipeline": "#2E75B6",
                "agent": "#E74C3C"
            },
            labels={
                "routed_to": "Routing Tier",
                "estimated_cost_usd": "Estimated Cost (USD)"
            },
            text_auto=".4f"
        )
        fig_cost.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            showlegend=False,
            xaxis_title="Tier",
            yaxis_title="Total Cost (USD)"
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        total_cost = sum(r["estimated_cost_usd"] for r in cost_data)
        total_interactions = sum(r["interaction_count"] for r in cost_data)
        if total_interactions > 0:
            st.metric(
                "Total Estimated Cost",
                f"${total_cost:.4f}",
                delta=f"${total_cost/total_interactions:.4f} per interaction avg"
            )
    else:
        st.info("No cost data available yet.")

st.markdown("---")

# ─────────────────────────────────────────────
# PANEL 4 — LATENCY DISTRIBUTION
# ─────────────────────────────────────────────

col_lat, col_feed = st.columns(2)

with col_lat:
    st.markdown('<div class="section-header">Latency Distribution</div>',
                unsafe_allow_html=True)

    if logs:
        latencies = [log["latency_ms"] for log in logs if log["latency_ms"]]
        tiers = [log["routed_to"] for log in logs if log["latency_ms"]]

        df_lat = pd.DataFrame({
            "latency_ms": latencies,
            "tier": tiers
        })

        fig_lat = px.box(
            df_lat,
            x="tier",
            y="latency_ms",
            color="tier",
            color_discrete_map={
                "faq_lookup": "#2ECC71",
                "faq_no_match": "#F39C12",
                "rag_pipeline": "#2E75B6",
                "agent": "#E74C3C"
            },
            labels={
                "latency_ms": "Latency (ms)",
                "tier": "Routing Tier"
            }
        )
        fig_lat.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        # P50/P95/P99 summary
        ld = metrics["latency_distribution"]
        cols = st.columns(3)
        cols[0].metric("P50", f"{ld.get('p50_ms', 0):,}ms")
        cols[1].metric("P95", f"{ld.get('p95_ms', 0):,}ms")
        cols[2].metric("P99", f"{ld.get('p99_ms', 0):,}ms")
    else:
        st.info("No latency data available yet.")

# ─────────────────────────────────────────────
# PANEL 5 — FEEDBACK DISTRIBUTION
# ─────────────────────────────────────────────

with col_feed:
    st.markdown('<div class="section-header">Customer Feedback</div>',
                unsafe_allow_html=True)

    feedback_logs = [
        log for log in logs
        if log["human_feedback_score"] is not None
    ]

    if feedback_logs:
        score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for log in feedback_logs:
            score = log["human_feedback_score"]
            if score in score_counts:
                score_counts[score] += 1

        df_feedback = pd.DataFrame({
            "score": list(score_counts.keys()),
            "count": list(score_counts.values()),
            "label": ["😡 1", "😞 2", "😐 3", "😊 4", "😍 5"]
        })

        fig_feed = px.bar(
            df_feedback,
            x="label",
            y="count",
            color="score",
            color_continuous_scale=[
                "#E74C3C", "#E67E22",
                "#F1C40F", "#2ECC71", "#27AE60"
            ],
            labels={"label": "Score", "count": "Interactions"}
        )
        fig_feed.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_feed, use_container_width=True)

        fs = metrics["feedback_summary"]
        st.metric(
            "Average Score",
            f"{fs.get('avg_score', 0)}/5",
            delta=f"From {fs.get('total_with_feedback', 0)} rated interactions"
        )
    else:
        st.info("No feedback data available yet.")

st.markdown("---")

# ─────────────────────────────────────────────
# PANEL 6 — RECENT INTERACTIONS LOG
# ─────────────────────────────────────────────

st.markdown('<div class="section-header">Recent Interactions</div>',
            unsafe_allow_html=True)

if logs:
    df_logs = pd.DataFrame(logs)

    # Clean up for display
    df_display = df_logs[[
        "log_id", "created_at", "routed_to",
        "latency_ms", "human_feedback_score",
        "escalated_to_human", "query_text"
    ]].copy()

    df_display.columns = [
        "ID", "Timestamp", "Tier",
        "Latency(ms)", "Feedback", "Escalated", "Query"
    ]

    df_display["Query"] = df_display["Query"].str[:60] + "..."
    df_display["Escalated"] = df_display["Escalated"].map(
        {0: "No", 1: "Yes ⚠️"}
    )
    df_display["Feedback"] = df_display["Feedback"].apply(
        lambda x: f"{x}/5" if x else "—"
    )

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn(width="small"),
            "Tier": st.column_config.TextColumn(width="medium"),
            "Latency(ms)": st.column_config.NumberColumn(width="small"),
            "Feedback": st.column_config.TextColumn(width="small"),
            "Escalated": st.column_config.TextColumn(width="small"),
        }
    )

    # Expandable detail view
    st.markdown("**View interaction detail:**")
    selected_id = st.selectbox(
        "Select log ID",
        options=[log["log_id"] for log in logs],
        label_visibility="collapsed"
    )

    selected_log = next(
        (log for log in logs if log["log_id"] == selected_id), None
    )
    if selected_log:
        with st.expander(f"Interaction {selected_id} — Full Detail", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Query:** {selected_log['query_text']}")
                st.markdown(f"**Tier:** {selected_log['routed_to']}")
                st.markdown(f"**Latency:** {selected_log['latency_ms']}ms")
                st.markdown(f"**Tokens:** {(selected_log['tokens_used_input'] or 0) + (selected_log['tokens_used_output'] or 0)}")
            with col_b:
                st.markdown(f"**Feedback:** {selected_log['human_feedback_score'] or 'Not rated'}")
                st.markdown(f"**Escalated:** {'Yes' if selected_log['escalated_to_human'] else 'No'}")
                tools = json.loads(selected_log["tools_called"] or "[]")
                if tools:
                    st