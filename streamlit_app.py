"""
Streamlit UI for Knowledge Intelligence System
Replicates the Flask HTML interface with Streamlit components.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json

# ==================== Configuration ====================

FLASK_API_URL = "http://localhost:8080"

st.set_page_config(
    page_title="Knowledge Intelligence System",
    page_icon="🧠",
    layout="centered"
)

# ==================== Custom CSS (matches style.css) ====================

st.markdown("""
<style>
    /* Container & typography */
    .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    
    /* Section cards */
    .section-card {
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 20px;
    }

    /* Hint text */
    .hint {
        color: #666;
        font-size: 0.85em;
        margin-bottom: 8px;
    }

    /* Response type badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
        margin-bottom: 10px;
        color: #fff;
    }
    .badge-credit_decision { background-color: #6f42c1; }
    .badge-credit_decision_error { background-color: #dc3545; }
    .badge-dual_store_search { background-color: #17a2b8; }
    .badge-vector_search { background-color: #17a2b8; }
    .badge-llm_only { background-color: #6c757d; }

    /* Decision Card */
    .decision-result {
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid;
        margin-top: 10px;
    }
    .decision-result.approved {
        background: #d4edda;
        border-left-color: #28a745;
    }
    .decision-result.declined {
        background: #f8d7da;
        border-left-color: #dc3545;
    }
    .decision-result.refer {
        background: #fff3cd;
        border-left-color: #ffc107;
    }

    .decision-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .decision-code {
        font-size: 1.3em;
        font-weight: 700;
    }
    .decision-desc {
        font-size: 1em;
        color: #555;
    }

    .detail-row {
        padding: 3px 0;
    }
    .risk-low { color: #28a745; font-weight: 600; }
    .risk-medium { color: #ffc107; font-weight: 600; }
    .risk-high { color: #dc3545; font-weight: 600; }

    .reason-codes ul {
        margin: 4px 0 0 0;
        padding-left: 20px;
    }
    .reason-codes li { padding: 2px 0; }
    .reason-codes code {
        background: rgba(0,0,0,0.08);
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.9em;
    }

    .request-id {
        margin-top: 10px;
        font-size: 0.8em;
        color: #888;
    }
    .available-customers {
        margin-top: 10px;
        font-size: 0.85em;
        color: #555;
    }

    /* Sources info */
    .sources-info {
        margin-top: 10px;
        padding: 8px 12px;
        background: #e9ecef;
        border-radius: 5px;
        font-size: 0.85em;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Helper Functions ====================

def render_badge(response_type: str) -> str:
    """Render a response type badge matching the Flask UI."""
    labels = {
        "credit_decision": "Credit Decision Engine",
        "credit_decision_error": "Credit Decision Error",
        "dual_store_search": "Vector Store + Knowledge Graph",
        "vector_search": "Vector Store Search",
        "llm_only": "LLM Response",
    }
    label = labels.get(response_type, response_type or "unknown")
    return f'<span class="badge badge-{response_type}">{label}</span>'


def render_decision_card(decision: dict, available_customers: list = None) -> str:
    """Render a credit decision card matching the Flask UI."""
    d_info = decision.get("decision", {})
    risk = decision.get("risk", {})
    reason_codes = decision.get("reasonCodes", [])
    header = decision.get("header", {})

    code = d_info.get("decisionCode", "UNKNOWN")
    desc = d_info.get("decisionDescription", "")
    risk_score = risk.get("riskScore", "N/A")
    risk_band = risk.get("riskBand", "UNKNOWN")

    if code == "APPROVED":
        card_class = "approved"
    elif code == "REFER":
        card_class = "refer"
    else:
        card_class = "declined"

    html = f'<div class="decision-result {card_class}">'
    html += f'<div class="decision-header">'
    html += f'<span class="decision-code">{code}</span>'
    html += f'<span class="decision-desc">{desc}</span>'
    html += f'</div>'

    html += f'<div class="decision-details">'
    html += f'<div class="detail-row"><strong>Risk Score:</strong> {risk_score}</div>'
    html += f'<div class="detail-row"><strong>Risk Band:</strong> <span class="risk-{risk_band.lower()}">{risk_band}</span></div>'

    if code == "APPROVED":
        credit_limit = d_info.get("creditLimit")
        interest_rate = d_info.get("interestRate")
        tenure = d_info.get("tenureMonths")
        cl_display = f"₹{credit_limit:,}" if credit_limit else "N/A"
        html += f'<div class="detail-row"><strong>Credit Limit:</strong> {cl_display}</div>'
        html += f'<div class="detail-row"><strong>Interest Rate:</strong> {interest_rate or "N/A"}%</div>'
        html += f'<div class="detail-row"><strong>Tenure:</strong> {tenure or "N/A"} months</div>'
    html += '</div>'

    html += '<div class="reason-codes"><strong>Reason Codes:</strong><ul>'
    for rc in reason_codes:
        html += f'<li><code>{rc["code"]}</code> — {rc["description"]}</li>'
    html += '</ul></div>'

    request_id = header.get("requestId", "N/A")
    html += f'<div class="request-id">Request: {request_id}</div>'
    html += '</div>'

    if available_customers:
        html += f'<div class="available-customers"><strong>Available customers:</strong> {", ".join(available_customers)}</div>'

    return html


# ==================== Session State ====================

if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "ingest_message" not in st.session_state:
    st.session_state.ingest_message = None
if "storage_backend" not in st.session_state:
    st.session_state.storage_backend = "local"
if "local_upload_dir" not in st.session_state:
    st.session_state.local_upload_dir = "uploaded_docs"


# ==================== Page Layout ====================

st.title("🧠 Knowledge Intelligence System")

# ---------- STEP 1: Upload Documents Section ----------
st.header("📁 Step 1: Upload Documents")
st.caption("Upload files to local directory and/or S3 bucket. No processing happens here.")

# --- Storage destination controls ---
st.subheader("Storage Destination")
storage_col1, storage_col2 = st.columns(2)

with storage_col1:
    storage_backend = st.selectbox(
        "Storage Backend",
        options=["local", "s3"],
        index=0 if st.session_state.storage_backend == "local" else 1,
        key="storage_backend_select",
        help="Choose where uploaded files are stored.",
    )
    st.session_state.storage_backend = storage_backend

with storage_col2:
    if storage_backend == "local":
        local_dir = st.text_input(
            "Local Folder Path",
            value=st.session_state.local_upload_dir,
            key="local_dir_input",
            help="Absolute or relative path for local file storage.",
        )
        st.session_state.local_upload_dir = local_dir
    else:
        st.info("Files will be uploaded to the S3 bucket configured on the server.")

uploaded_files = st.file_uploader(
    "Choose files to upload",
    type=["txt", "pdf", "docx", "json", "xlsx"],
    accept_multiple_files=True,
    key="file_uploader"
)

if st.button("Upload", key="upload_btn"):
    if not uploaded_files:
        st.warning("Please select at least one file.")
    else:
        success_files = []
        failed_files = []
        for uf in uploaded_files:
            try:
                files = {"file": (uf.name, uf.getvalue(), uf.type or "application/octet-stream")}
                form_data = {"storage_backend": st.session_state.storage_backend}
                if st.session_state.storage_backend == "local":
                    form_data["local_dir"] = st.session_state.local_upload_dir
                resp = requests.post(
                    f"{FLASK_API_URL}/upload", files=files, data=form_data, timeout=60
                )
                data = resp.json()
                if resp.ok:
                    success_files.append(uf.name)
                else:
                    failed_files.append(f"**{uf.name}** — {data.get('error', 'Upload failed')}")
            except requests.ConnectionError:
                failed_files.append(f"**{uf.name}** — Cannot connect to Flask API at {FLASK_API_URL}")
            except Exception as e:
                failed_files.append(f"**{uf.name}** — {str(e)}")

        # Show Upload Successfully popup
        if success_files:
            backend_label = "Local Storage" if st.session_state.storage_backend == "local" else "S3 Bucket"
            st.success(
                f"Upload Successfully!\n\n"
                f"{len(success_files)} file(s) stored in **{backend_label}**: "
                + ", ".join(success_files)
            )
        if failed_files:
            st.error("Some files failed to upload:\n\n" + "\n\n".join(failed_files))

st.divider()

# ---------- STEP 2-6: Ingestion Pipeline Section ----------
st.header("⚙️ Step 2-6: Ingestion Pipeline")
st.caption(
    "Run the full ingestion pipeline on all uploaded files:\n"
    "**Step 2** Ingestion → **Step 3** AI Extraction → **Step 4** Normalization → "
    "**Step 5** Vector Store (PDF/DOCX/TXT/XLSX) → **Step 6** Knowledge Graph (JSON)"
)

if st.button("🚀 Start Ingestion", key="ingest_btn", type="primary"):
    with st.spinner("Running ingestion pipeline (Steps 2-6)..."):
        try:
            payload = {
                "storage_backend": st.session_state.storage_backend,
            }
            if st.session_state.storage_backend == "local":
                payload["local_dir"] = st.session_state.local_upload_dir

            resp = requests.post(
                f"{FLASK_API_URL}/api/ingest",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300,
            )
            data = resp.json()
            if resp.ok:
                steps = data.get("pipeline_steps", {})
                parts = [f"✅ **Pipeline completed** — {data.get('message', '')}"]

                # Step 2: Ingestion
                s2 = steps.get("step2_ingestion", [])
                if s2:
                    ok_count = sum(1 for r in s2 if 'error' not in r)
                    parts.append(f"\n**Step 2 — Ingestion:** {ok_count}/{len(s2)} files ingested")

                # Step 3: Extraction
                s3 = steps.get("step3_extraction", {})
                if s3:
                    parts.append(
                        f"**Step 3 — AI Extraction:** {s3.get('total_entities', 0)} entities extracted"
                    )
                    by_type = s3.get("entities_by_type", {})
                    if by_type:
                        parts.append("  " + ", ".join(f"{k}: {v}" for k, v in by_type.items()))
                    errors = s3.get("errors", [])
                    warnings = s3.get("warnings", [])
                    if errors:
                        parts.append("  Errors:")
                        parts.extend(f"    - {error}" for error in errors)
                    if warnings:
                        parts.append("  Warnings:")
                        parts.extend(f"    - {warning}" for warning in warnings)

                # Step 4: Normalization
                s4 = steps.get("step4_normalization", {})
                if s4:
                    parts.append(
                        f"**Step 4 — Normalization:** {s4.get('total_normalized', 0)} entities "
                        f"({s4.get('duplicates_removed', 0)} duplicates removed)"
                    )

                # Step 5: Vector Store
                s5 = steps.get("step5_vector_store", [])
                if s5:
                    total_chunks = sum(r.get("chunks_stored", 0) for r in s5)
                    filenames = [r["filename"] for r in s5 if "filename" in r]
                    parts.append(
                        f"**Step 5 — Vector Store:** {len(s5)} files, {total_chunks} chunks"
                    )
                    if filenames:
                        parts.append("  Files: " + ", ".join(filenames))

                # Step 6: Knowledge Graph
                s6 = steps.get("step6_knowledge_graph", [])
                if s6:
                    kg_files = [r for r in s6 if "filename" in r]
                    parts.append(
                        f"**Step 6 — Knowledge Graph:** {len(kg_files)} JSON files ingested"
                    )
                    for r in kg_files:
                        parts.append(f"  • {r['filename']}: {r.get('total_nodes', '?')} nodes")

                # Step 7: Validation
                s7 = steps.get("step7_validation", {})
                if s7:
                    score = s7.get("validation_score", 0)
                    parts.append(
                        f"**Step 7 — Validation:** Score {score:.2f} | "
                        f"Conflicts: {s7.get('conflicts', 0)} | "
                        f"Gaps: {s7.get('gaps', 0)} | "
                        f"Inconsistencies: {s7.get('inconsistencies', 0)}"
                    )
                    if s7.get("is_valid"):
                        parts.append("  ✅ No high-severity conflicts")
                    else:
                        parts.append("  ⚠️ High-severity conflicts detected — review needed")

                # Step 8: Traceability
                s8 = steps.get("step8_traceability", {})
                if s8:
                    parts.append(
                        f"**Step 8 — Traceability:** {s8.get('total_traces', 0)} traces | "
                        f"Avg confidence: {s8.get('average_confidence', 0):.3f}"
                    )

                st.session_state.ingest_message = "\n\n".join(parts)
            else:
                st.session_state.ingest_message = f"❌ {data.get('error', 'Ingestion failed')}"
        except requests.ConnectionError:
            st.session_state.ingest_message = f"❌ Cannot connect to Flask API at {FLASK_API_URL}"
        except Exception as e:
            st.session_state.ingest_message = f"❌ {str(e)}"

if st.session_state.ingest_message:
    st.markdown(st.session_state.ingest_message)

st.divider()

# ---------- STEP 7: Validation Section ----------
st.header("🔍 Step 7: Validation")
st.caption("Detect conflicts, gaps, and inconsistencies in extracted knowledge.")

if "validation_result" not in st.session_state:
    st.session_state.validation_result = None

if st.button("Run Validation", key="validate_btn"):
    with st.spinner("Running validation..."):
        try:
            resp = requests.post(f"{FLASK_API_URL}/api/validate", timeout=120)
            data = resp.json()
            if resp.ok:
                st.session_state.validation_result = data
            else:
                st.session_state.validation_result = {"error": data.get("error", "Validation failed")}
        except requests.ConnectionError:
            st.session_state.validation_result = {"error": f"Cannot connect to Flask API at {FLASK_API_URL}"}
        except Exception as e:
            st.session_state.validation_result = {"error": str(e)}

if st.session_state.validation_result:
    vr = st.session_state.validation_result
    if "error" in vr:
        st.warning(vr["error"])
    else:
        score = vr.get("validation_score", 0)
        is_valid = vr.get("is_valid", False)
        conflicts = vr.get("conflicts", [])
        gaps = vr.get("gaps", [])
        inconsistencies = vr.get("inconsistencies", [])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score", f"{score:.2f}")
        col2.metric("Conflicts", len(conflicts))
        col3.metric("Gaps", len(gaps))
        col4.metric("Inconsistencies", len(inconsistencies))

        if is_valid:
            st.success("Knowledge base is valid — no high-severity conflicts.")
        else:
            st.error("High-severity conflicts detected — human review required.")

        if conflicts:
            with st.expander(f"Conflicts ({len(conflicts)})"):
                for c in conflicts:
                    severity = c.get("severity", "medium")
                    icon = "🔴" if severity == "high" else ("🟡" if severity == "medium" else "🟢")
                    st.markdown(f"{icon} **[{severity.upper()}]** {c.get('description', '')}")
        if gaps:
            with st.expander(f"Gaps ({len(gaps)})"):
                for g in gaps:
                    st.markdown(f"⚠️ {g}")
        if inconsistencies:
            with st.expander(f"Inconsistencies ({len(inconsistencies)})"):
                for inc in inconsistencies:
                    st.markdown(f"⚠️ {inc}")

st.divider()

# ---------- STEP 8: Traceability Section ----------
st.header("🔗 Step 8: Traceability")
st.caption("View source, confidence scores, and version info for all extracted entities.")

if "traceability_result" not in st.session_state:
    st.session_state.traceability_result = None

if st.button("View Traceability", key="trace_btn"):
    with st.spinner("Fetching traceability..."):
        try:
            resp = requests.get(f"{FLASK_API_URL}/api/traceability", timeout=60)
            data = resp.json()
            if resp.ok:
                st.session_state.traceability_result = data
            else:
                st.session_state.traceability_result = {"error": data.get("error", "Failed")}
        except requests.ConnectionError:
            st.session_state.traceability_result = {"error": f"Cannot connect to Flask API at {FLASK_API_URL}"}
        except Exception as e:
            st.session_state.traceability_result = {"error": str(e)}

if st.session_state.traceability_result:
    tr = st.session_state.traceability_result
    if "error" in tr and "traces" not in tr:
        st.warning(tr["error"])
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total Traces", tr.get("total_traces", 0))
        col2.metric("Avg Confidence", f"{tr.get('average_confidence', 0):.3f}")

        dist = tr.get("confidence_distribution", {})
        if dist:
            st.markdown("**Confidence Distribution:** " + ", ".join(f"{k}: {v}" for k, v in dist.items()))

        traces = tr.get("traces", [])
        if traces:
            with st.expander(f"All traces ({len(traces)})"):
                for t in traces[:50]:
                    conf = t.get("confidence_score", 0)
                    level = t.get("confidence_level", "")
                    icon = "🟢" if level == "high" else ("🟡" if level == "medium" else "🔴")
                    st.markdown(
                        f"{icon} **{t.get('entity_id', '')}** ({t.get('entity_type', '')}) "
                        f"— Source: `{t.get('source', '')}` | Confidence: {conf:.2f} | "
                        f"Method: {t.get('extraction_method', '')} | Version: {t.get('version', '1.0')}"
                    )
                if len(traces) > 50:
                    st.caption(f"Showing first 50 of {len(traces)} traces.")

st.divider()

# ---------- STEP 9: Output Generation Section ----------
st.header("📝 Step 9: Output Generation")
st.caption("Generate documents (requirements, design, business rules) from extracted knowledge.")

if "generated_doc" not in st.session_state:
    st.session_state.generated_doc = None

gen_col1, gen_col2, gen_col3 = st.columns(3)

with gen_col1:
    if st.button("Generate Requirements", key="gen_req_btn"):
        with st.spinner("Generating requirements document..."):
            try:
                resp = requests.post(f"{FLASK_API_URL}/api/generate/requirements", timeout=300)
                data = resp.json()
                if resp.ok:
                    st.session_state.generated_doc = data
                else:
                    st.session_state.generated_doc = {"error": data.get("error", "Generation failed")}
            except requests.ConnectionError:
                st.session_state.generated_doc = {"error": f"Cannot connect to Flask API at {FLASK_API_URL}"}
            except Exception as e:
                st.session_state.generated_doc = {"error": str(e)}

with gen_col2:
    if st.button("Generate Design", key="gen_design_btn"):
        with st.spinner("Generating design document..."):
            try:
                resp = requests.post(f"{FLASK_API_URL}/api/generate/design", timeout=300)
                data = resp.json()
                if resp.ok:
                    st.session_state.generated_doc = data
                else:
                    st.session_state.generated_doc = {"error": data.get("error", "Generation failed")}
            except requests.ConnectionError:
                st.session_state.generated_doc = {"error": f"Cannot connect to Flask API at {FLASK_API_URL}"}
            except Exception as e:
                st.session_state.generated_doc = {"error": str(e)}

with gen_col3:
    if st.button("Generate Rules", key="gen_rules_btn"):
        with st.spinner("Generating business rules document..."):
            try:
                resp = requests.post(f"{FLASK_API_URL}/api/generate/rules", timeout=300)
                data = resp.json()
                if resp.ok:
                    st.session_state.generated_doc = data
                else:
                    st.session_state.generated_doc = {"error": data.get("error", "Generation failed")}
            except requests.ConnectionError:
                st.session_state.generated_doc = {"error": f"Cannot connect to Flask API at {FLASK_API_URL}"}
            except Exception as e:
                st.session_state.generated_doc = {"error": str(e)}

if st.session_state.generated_doc:
    gd = st.session_state.generated_doc
    if "error" in gd and "content" not in gd:
        st.warning(gd["error"])
    else:
        st.subheader(gd.get("title", "Generated Document"))
        st.caption(
            f"Type: {gd.get('type', 'N/A')} | "
            f"Source entities: {gd.get('source_entity_count', 0)} | "
            f"Generated: {gd.get('generated_at', 'N/A')}"
        )
        if gd.get("requires_review"):
            st.info("⚠️ This document requires human review before use.")
        st.markdown(gd.get("content", ""))

st.divider()

# ---------- STEP 10: Observability Section ----------
st.header("📊 Step 10: Observability")
st.caption("Track accuracy, confidence, and system changes.")

if "observability_data" not in st.session_state:
    st.session_state.observability_data = None

obs_col1, obs_col2 = st.columns(2)

with obs_col1:
    if st.button("View Metrics", key="metrics_btn"):
        with st.spinner("Fetching metrics..."):
            try:
                resp = requests.get(f"{FLASK_API_URL}/api/metrics", timeout=30)
                data = resp.json()
                if resp.ok:
                    st.session_state.observability_data = {"metrics": data}
                else:
                    st.session_state.observability_data = {"error": data.get("error", "Failed")}
            except requests.ConnectionError:
                st.session_state.observability_data = {"error": f"Cannot connect to Flask API at {FLASK_API_URL}"}
            except Exception as e:
                st.session_state.observability_data = {"error": str(e)}

with obs_col2:
    if st.button("View Event Log", key="events_btn"):
        with st.spinner("Fetching events..."):
            try:
                resp = requests.get(f"{FLASK_API_URL}/api/events?limit=50", timeout=30)
                data = resp.json()
                if resp.ok:
                    st.session_state.observability_data = {"events": data}
                else:
                    st.session_state.observability_data = {"error": data.get("error", "Failed")}
            except requests.ConnectionError:
                st.session_state.observability_data = {"error": f"Cannot connect to Flask API at {FLASK_API_URL}"}
            except Exception as e:
                st.session_state.observability_data = {"error": str(e)}

if st.session_state.observability_data:
    od = st.session_state.observability_data
    if "error" in od:
        st.warning(od["error"])
    elif "metrics" in od:
        metrics = od["metrics"]
        st.json(metrics)
    elif "events" in od:
        events = od["events"].get("events", [])
        st.write(f"**{len(events)} recent events**")
        for ev in events[-20:]:
            st.markdown(
                f"- `{ev.get('timestamp', '')}` **{ev.get('type', '')}** — "
                f"{json.dumps(ev.get('details', {}), default=str)}"
            )

st.divider()

# ---------- STEP 11: Feedback Loop Section ----------
st.header("💬 Step 11: Feedback Loop")
st.caption("Submit feedback to improve the system continuously. Human review improves accuracy over time.")

if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = None

with st.form("feedback_form", clear_on_submit=True):
    fb_entity_id = st.text_input("Entity ID", placeholder="e.g. norm_0")
    fb_type = st.selectbox("Feedback Type", ["correction", "suggestion", "validation", "edit"])
    fb_original = st.text_area("Original Text", placeholder="The original extracted text")
    fb_corrected = st.text_area("Corrected Text (optional)", placeholder="Your correction")
    fb_notes = st.text_input("Notes (optional)", placeholder="Additional comments")
    fb_submitted = st.form_submit_button("Submit Feedback")

    if fb_submitted:
        if not fb_entity_id or not fb_original:
            st.session_state.feedback_message = ("warning", "Entity ID and Original Text are required.")
        else:
            try:
                payload = {
                    "entity_id": fb_entity_id,
                    "type": fb_type,
                    "original_text": fb_original,
                    "corrected_text": fb_corrected or None,
                    "notes": fb_notes or None,
                }
                resp = requests.post(
                    f"{FLASK_API_URL}/api/feedback",
                    json=payload,
                    timeout=30,
                )
                data = resp.json()
                if resp.ok:
                    st.session_state.feedback_message = (
                        "success",
                        f"Feedback submitted — ID: {data.get('feedback_id', 'N/A')}",
                    )
                else:
                    st.session_state.feedback_message = ("error", data.get("error", "Submission failed"))
            except requests.ConnectionError:
                st.session_state.feedback_message = ("error", f"Cannot connect to Flask API at {FLASK_API_URL}")
            except Exception as e:
                st.session_state.feedback_message = ("error", str(e))

if st.session_state.feedback_message:
    level, msg = st.session_state.feedback_message
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.error(msg)

st.divider()

# ---------- Ask Questions Section ----------
st.header("Ask Questions")
st.markdown(
    '<p class="hint">Try: "What is the credit decision for customer?" or '
    '"credit decision" to evaluate customers from the stores.</p>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Enter your question",
    placeholder="Enter your question...",
    height=100,
    label_visibility="collapsed",
    key="question_input"
)

if st.button("Ask", key="ask_btn"):
    if not question or not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            try:
                resp = requests.post(
                    f"{FLASK_API_URL}/query",
                    json={"question": question.strip()},
                    headers={"Content-Type": "application/json"},
                    timeout=120,
                )
                st.session_state.query_result = resp.json()
            except requests.ConnectionError:
                st.session_state.query_result = {
                    "error": f"Cannot connect to Flask API at {FLASK_API_URL}. Is the server running?"
                }
            except Exception as e:
                st.session_state.query_result = {"error": str(e)}

st.divider()

# ---------- Response Section ----------
st.header("Response")

result = st.session_state.query_result

if result:
    # Error case
    if "error" in result and result.get("type") is None:
        st.error(result["error"])
    else:
        resp_type = result.get("type", "llm_only")

        # Badge
        st.markdown(render_badge(resp_type), unsafe_allow_html=True)

        # Text response
        response_text = result.get("response", "")
        if response_text:
            st.markdown(
                f'<div style="min-height:50px; padding:10px; border:1px solid #ddd; '
                f'border-radius:5px; white-space:pre-wrap; word-wrap:break-word;">{response_text}</div>',
                unsafe_allow_html=True,
            )

        # Credit decision card
        if resp_type == "credit_decision" and result.get("decision"):
            card_html = render_decision_card(
                result["decision"],
                result.get("available_customers"),
            )
            st.markdown(card_html, unsafe_allow_html=True)

        # Credit decision error with available customers
        if resp_type == "credit_decision_error" and result.get("available_customers"):
            customers = result["available_customers"]
            st.info(f"**Available customers:** {', '.join(customers)}")

        # Sources info (vector search / dual store)
        if resp_type in ("vector_search", "dual_store_search") and result.get("sources"):
            unique_sources = list(dict.fromkeys(result["sources"]))
            vs_count = result.get("vector_store_chunks", result.get("chunks_found", 0))
            kg_count = result.get("knowledge_graph_chunks", 0)
            source_parts = [f"<strong>Sources:</strong> {', '.join(unique_sources)}"]
            if resp_type == "dual_store_search":
                source_parts.append(
                    f"(Vector Store: {vs_count} chunks, Knowledge Graph: {kg_count} chunks)"
                )
            else:
                source_parts.append(f"({result.get('chunks_found', 0)} chunks matched)")
            st.markdown(
                f'<div class="sources-info">{" ".join(source_parts)}</div>',
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        '<div style="min-height:50px; padding:10px; border:1px solid #ddd; '
        'border-radius:5px; color:#999;">Responses will appear here...</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------- Sidebar: System Info & Quick Actions ----------
with st.sidebar:
    st.header("⚙️ System")

    # Status check
    if st.button("Check API Status"):
        try:
            resp = requests.get(f"{FLASK_API_URL}/api/status", timeout=10)
            if resp.ok:
                status = resp.json()
                st.success(f"API: {status.get('status', 'unknown')}")
                config = status.get("config", {})
                st.json(config)
                vs_stats = status.get("vector_store_stats", {})
                if vs_stats:
                    st.metric("Vector Store Docs", vs_stats.get("total_documents", 0))
                kg_stats = status.get("knowledge_graph_stats", {})
                if kg_stats:
                    st.metric("KG Nodes", kg_stats.get("total_nodes", 0))
                    st.metric("KG Edges", kg_stats.get("total_edges", 0))
            else:
                st.error(f"API returned {resp.status_code}")
        except requests.ConnectionError:
            st.error(f"Cannot connect to {FLASK_API_URL}")

    st.divider()

    # Ingest sample data
    st.subheader("Quick Actions")
    if st.button("Ingest Sample Zoot Data"):
        with st.spinner("Ingesting..."):
            try:
                resp = requests.post(f"{FLASK_API_URL}/api/ingest/sample-data", timeout=120)
                data = resp.json()
                if resp.ok:
                    st.success(data.get("message", "Done"))
                    kg_docs = data.get("knowledge_graph_documents", [])
                    if kg_docs:
                        st.write("**Knowledge Graph:**")
                        for d in kg_docs:
                            st.write(f"  • {d['filename']} ({d['total_nodes']} nodes)")
                else:
                    st.error(data.get("error", "Failed"))
            except requests.ConnectionError:
                st.error("Cannot connect to API")

    # List customers
    if st.button("List Customers"):
        try:
            resp = requests.get(f"{FLASK_API_URL}/api/decision/customers", timeout=10)
            data = resp.json()
            customers = data.get("customers", [])
            if customers:
                st.write(f"**{len(customers)} customer(s):**")
                for c in customers:
                    st.code(c)
            else:
                st.info("No customers found. Ingest data first.")
        except requests.ConnectionError:
            st.error("Cannot connect to API")

    # Run pipeline
    if st.button("Run Full Pipeline"):
        with st.spinner("Running pipeline..."):
            try:
                resp = requests.post(f"{FLASK_API_URL}/api/pipeline/run", timeout=300)
                data = resp.json()
                if resp.ok:
                    st.success("Pipeline completed")
                    st.json(data.get("steps", {}))
                else:
                    st.error(data.get("error", "Pipeline failed"))
            except requests.ConnectionError:
                st.error("Cannot connect to API")

    st.divider()
    st.caption(f"API: {FLASK_API_URL}")

    # Storage config info
    st.divider()
    st.subheader("Storage Config")
    try:
        resp = requests.get(f"{FLASK_API_URL}/api/storage/config", timeout=5)
        if resp.ok:
            scfg = resp.json()
            st.write(f"**Backend:** {scfg.get('storage_backend', 'N/A')}")
            if scfg.get("storage_backend") == "local":
                st.write(f"**Local Dir:** {scfg.get('local_upload_dir', 'N/A')}")
            else:
                st.write(f"**S3 Bucket:** {scfg.get('aws_bucket', 'N/A')}")
                st.write(f"**AWS Region:** {scfg.get('aws_region', 'N/A')}")
    except Exception:
        st.caption("Could not fetch storage config.")
