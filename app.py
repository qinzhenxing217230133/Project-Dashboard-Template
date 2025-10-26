# app.py
import os
import datetime as dt
import traceback
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from pymongo import MongoClient

# ───────────────────────────────
# Env & constants
# ───────────────────────────────
load_dotenv()

PG_SCHEMA = os.getenv("PG_SCHEMA", "public")  # default schema

def qualify(sql: str) -> str:
    """
    Replace occurrences of {S}.<table> with <schema>.<table>.
    Keeps your saved queries readable while honoring custom schemas.
    """
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

def require_env():
    """
    Basic env validation with friendly UI feedback.
    """
    problems = []
    if not os.getenv("PG_URI"):
        problems.append("PG_URI 未设置，已使用默认本地连接。")
    if not os.getenv("MONGO_URI"):
        problems.append("MONGO_URI 未设置，已使用默认本地连接。")
    if not os.getenv("MONGO_DB"):
        problems.append("MONGO_DB 未设置，已使用 eldercare。")

    if problems:
        for p in problems:
            st.warning(p, icon="⚠️")

require_env()

# ───────────────────────────────
# CONFIG: Postgres and Mongo Queries
# ───────────────────────────────
CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv(
            "PG_URI",
            "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
        ),
        "queries": {
            # DOCTORS
            "Doctor: patients under my care (table)": {
                "sql": """
                    SELECT p.patient_id, p.name AS patient, p.age, p.room_no
                    FROM {S}.patients p
                    WHERE p.doctor_id = :doctor_id 
                    ORDER BY p.name;
                """,
                "chart": {"type": "table"},
                "tags": ["doctor"],
                "params": ["doctor_id"]
            },
            "Doctor: most recent treatment per my patient (table)": {
                "sql": """
                    SELECT p.name AS patient,
                           (SELECT MAX(t.treatment_time)
                              FROM {S}.treatments t
                              WHERE t.patient_id = p.patient_id) AS last_treatment
                    FROM {S}.patients p
                    WHERE p.doctor_id = :doctor_id
                    ORDER BY last_treatment DESC NULLS LAST;
                """,
                "chart": {"type": "table"},
                "tags": ["doctor"],
                "params": ["doctor_id"]
            },
            "Doctor: high-risk (age > threshold) under my care (bar)": {
                "sql": """
                    SELECT p.name AS patient, p.age
                    FROM {S}.patients p
                    WHERE p.doctor_id = :doctor_id
                      AND p.age > :age_threshold
                    ORDER BY p.age DESC;
                """,
                "chart": {"type": "bar", "x": "patient", "y": "age"},
                "tags": ["doctor"],
                "params": ["doctor_id", "age_threshold"]
            },
            "Doctor: patients with NO treatment today (table)": {
                "sql": """
                    SELECT p.name, p.room_no
                    FROM {S}.patients p
                    WHERE p.doctor_id = :doctor_id
                      AND NOT EXISTS (
                        SELECT 1
                        FROM {S}.treatments t
                        WHERE t.patient_id = p.patient_id
                          AND t.treatment_time::date = CURRENT_DATE
                      );
                """,
                "chart": {"type": "table"},
                "tags": ["doctor"],
                "params": ["doctor_id"]
            },
            "Doctor: treatments by type for my patients (bar)": {
                "sql": """
                    SELECT t.treatment_type, COUNT(*)::int AS times_given
                    FROM {S}.treatments t
                    JOIN {S}.patients p ON p.patient_id = t.patient_id
                    WHERE p.doctor_id = :doctor_id
                    GROUP BY t.treatment_type
                    ORDER BY times_given DESC;
                """,
                "chart": {"type": "bar", "x": "treatment_type", "y": "times_given"},
                "tags": ["doctor"],
                "params": ["doctor_id"]
            },

            # NURSES
            "Nurse: today’s tasks (treatments to administer) (table)": {
                "sql": """
                    SELECT p.name AS patient, t.treatment_type, t.treatment_time
                    FROM {S}.treatments t
                    JOIN {S}.patients p ON t.patient_id = p.patient_id
                    WHERE t.nurse_id = :nurse_id
                      AND t.treatment_time::date = CURRENT_DATE
                    ORDER BY t.treatment_time;
                """,
                "chart": {"type": "table"},
                "tags": ["nurse"],
                "params": ["nurse_id"]
            },
            "Nurse: patients with NO treatment yet today (table)": {
                "sql": """
                    SELECT p.name, p.room_no
                    FROM {S}.patients p
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {S}.treatments t
                        WHERE t.patient_id = p.patient_id
                          AND t.treatment_time::date = CURRENT_DATE
                    )
                    ORDER BY p.room_no, p.name;
                """,
                "chart": {"type": "table"},
                "tags": ["nurse"]
            },
            "Nurse: medicines running low (bar)": {
                "sql": """
                    SELECT m.name, m.quantity
                    FROM {S}.medicine_stock m
                    WHERE m.quantity < :med_low_threshold
                    ORDER BY m.quantity ASC;
                """,
                "chart": {"type": "bar", "x": "name", "y": "quantity"},
                "tags": ["nurse"],
                "params": ["med_low_threshold"]
            },

            # PHARMACISTS
            "Pharmacist: medicines to reorder (bar)": {
                "sql": """
                    SELECT m.name, m.quantity
                    FROM {S}.medicine_stock m
                    WHERE m.quantity < :reorder_threshold
                    ORDER BY m.quantity ASC;
                """,
                "chart": {"type": "bar", "x": "name", "y": "quantity"},
                "tags": ["pharmacist"],
                "params": ["reorder_threshold"]
            },
            "Pharmacist: top 5 medicines this month (bar)": {
                "sql": """
                    SELECT t.treatment_type AS medicine, COUNT(*)::int AS times_given
                    FROM {S}.treatments t
                    WHERE t.treatment_time >= date_trunc('month', CURRENT_DATE)
                    GROUP BY t.treatment_type
                    ORDER BY times_given DESC
                    LIMIT 5;
                """,
                "chart": {"type": "bar", "x": "medicine", "y": "times_given"},
                "tags": ["pharmacist"]
            },
            "Pharmacist: which nurse gave most medicines today (table)": {
                "sql": """
                    SELECT n.name, COUNT(t.treatment_id)::int AS total
                    FROM {S}.nurses n
                    JOIN {S}.treatments t ON t.nurse_id = n.nurse_id
                    WHERE t.treatment_time::date = CURRENT_DATE
                    GROUP BY n.name
                    ORDER BY total DESC
                    LIMIT 1;
                """,
                "chart": {"type": "table"},
                "tags": ["pharmacist"]
            },
            "Pharmacist: medicines unused in last N days (table)": {
                "sql": """
                    SELECT m.name
                    FROM {S}.medicine_stock m
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {S}.treatments t
                        WHERE t.treatment_type = m.name
                          AND t.treatment_time >= NOW() - (:days || ' days')::interval
                    )
                    ORDER BY m.name;
                """,
                "chart": {"type": "table"},
                "tags": ["pharmacist"],
                "params": ["days"]
            },

            # FAMILY/GUARDIANS
            "Family: last treatment for my relative (table)": {
                "sql": """
                    SELECT t.treatment_type, t.treatment_time, n.name AS nurse
                    FROM {S}.treatments t
                    JOIN {S}.patients p ON t.patient_id = p.patient_id
                    LEFT JOIN {S}.nurses n ON t.nurse_id = n.nurse_id
                    WHERE p.name = :patient_name
                    ORDER BY t.treatment_time DESC
                    LIMIT 1;
                """,
                "chart": {"type": "table"},
                "tags": ["guardian"],
                "params": ["patient_name"]
            },
            "Family: which doctor is assigned to my relative? (table)": {
                "sql": """
                    SELECT p.name AS patient, d.name AS doctor, d.specialization
                    FROM {S}.patients p
                    JOIN {S}.doctors d ON p.doctor_id = d.doctor_id
                    WHERE p.name = :patient_name;
                """,
                "chart": {"type": "table"},
                "tags": ["guardian"],
                "params": ["patient_name"]
            },
            "Family: total treatments this month for my relative (table)": {
                "sql": """
                    SELECT COUNT(*)::int AS treatments_this_month
                    FROM {S}.treatments t
                    JOIN {S}.patients p ON t.patient_id = p.patient_id
                    WHERE p.name = :patient_name
                      AND t.treatment_time >= date_trunc('month', CURRENT_DATE);
                """,
                "chart": {"type": "table"},
                "tags": ["guardian"],
                "params": ["patient_name"]
            },

            # MANAGERS
            "Mgr: total patients & average age (table)": {
                "sql": """
                    SELECT COUNT(*)::int AS total_patients, AVG(age)::numeric(10,1) AS avg_age
                    FROM {S}.patients;
                """,
                "chart": {"type": "table"},
                "tags": ["manager"]
            },
            "Mgr: patients per doctor (bar)": {
                "sql": """
                    SELECT d.name AS doctor, COUNT(*)::int AS num_patients
                    FROM {S}.doctors d
                    LEFT JOIN {S}.patients p ON d.doctor_id = p.doctor_id
                    GROUP BY d.name
                    ORDER BY num_patients DESC;
                """,
                "chart": {"type": "bar", "x": "doctor", "y": "num_patients"},
                "tags": ["manager"]
            },
            "Mgr: treatments in last N days (table)": {
                "sql": """
                    SELECT COUNT(*)::int AS total_treatments
                    FROM {S}.treatments
                    WHERE treatment_time >= NOW() - (:days || ' days')::interval;
                """,
                "chart": {"type": "table"},
                "tags": ["manager"],
                "params": ["days"]
            },
            "Mgr: rooms currently occupied (table)": {
                "sql": """
                    SELECT DISTINCT p.room_no
                    FROM {S}.patients p
                    ORDER BY p.room_no;
                """,
                "chart": {"type": "table"},
                "tags": ["manager"]
            },
            "Mgr: doctor with oldest patients (table)": {
                "sql": """
                    SELECT d.name, MAX(p.age) AS oldest_patient_age
                    FROM {S}.doctors d
                    JOIN {S}.patients p ON d.doctor_id = p.doctor_id
                    GROUP BY d.name
                    ORDER BY oldest_patient_age DESC
                    LIMIT 1;
                """,
                "chart": {"type": "table"},
                "tags": ["manager"]
            }
        }
    },

    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "db_name": os.getenv("MONGO_DB", "eldercare"),
        "queries": {
            "TS: Hourly avg heart rate (resident 501, last 24h)": {
                "collection": "bracelet_readings_ts",
                "aggregate": [
                    {"$match": {
                        "meta.resident_id": 501,
                        "ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}
                    }},
                    {"$project": {
                        "hour": {"$dateTrunc": {"date": "$ts", "unit": "hour"}},
                        "hr": "$heart_rate_bpm"
                    }},
                    {"$group": {"_id": "$hour", "avg_hr": {"$avg": "$hr"}, "n": {"$count": {}}}},
                    {"$sort": {"_id": 1}}
                ],
                "chart": {"type": "line", "x": "_id", "y": "avg_hr"}
            },

            "TS: Exceedance counts (SpO2 < 92, last 7 days) by resident": {
                "collection": "bracelet_readings_ts",
                "aggregate": [
                    {"$match": {
                        "ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=7)},
                        "spo2_pct": {"$lt": 92}
                    }},
                    {"$group": {"_id": "$meta.resident_id", "hits": {"$count": {}}}},
                    {"$sort": {"hits": -1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "hits"}
            },

            "Telemetry: Latest reading per device": {
                "collection": "bracelet_data",
                "aggregate": [
                    {"$sort": {"ts": -1, "_id": -1}},
                    {"$group": {"_id": "$device_id", "doc": {"$first": "$$ROOT"}}},
                    {"$replaceRoot": {"newRoot": "$doc"}},
                    {"$project": {
                        "_id": 0, "device_id": 1, "resident_id": 1, "ts": 1,
                        "hr": "$metrics.heart_rate_bpm", "spo2": "$metrics.spo2_pct",
                        "status": 1
                    }}
                ],
                "chart": {"type": "table"}
            },

            "Telemetry: Battery status distribution": {
                "collection": "bracelet_data",
                "aggregate": [
                    {"$project": {
                        "battery": {"$ifNull": ["$battery_pct", None]},
                        "bucket": {
                            "$switch": {
                                "branches": [
                                    {"case": {"$gte": ["$battery_pct", 80]}, "then": "80–100"},
                                    {"case": {"$gte": ["$battery_pct", 60]}, "then": "60–79"},
                                    {"case": {"$gte": ["$battery_pct", 40]}, "then": "40–59"},
                                    {"case": {"$gte": ["$battery_pct", 20]}, "then": "20–39"},
                                ],
                                "default": "<20 or null"
                            }
                        }
                    }},
                    {"$group": {"_id": "$bucket", "cnt": {"$count": {}}}},
                    {"$sort": {"cnt": -1}}
                ],
                "chart": {"type": "pie", "names": "_id", "values": "cnt"}
            },

            "TS Treemap: readings count by resident and device (last 24h)": {
                "collection": "bracelet_readings_ts",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": {"resident": "$meta.resident_id", "device": "$meta.device_id"}, "cnt": {"$count": {}}}},
                    {"$project": {"resident": "$_id.resident", "device": "$_id.device", "cnt": 1, "_id": 0}}
                ],
                "chart": {"type": "treemap", "path": ["resident", "device"], "values": "cnt"}
            }
        }
    }
}

# ───────────────────────────────
# Streamlit page setup
# ───────────────────────────────
st.set_page_config(page_title="Old-Age Home DB Dashboard", layout="wide")
st.title("Old-Age Home | Mini Dashboard (Postgres + MongoDB)")

def metric_row(metrics: dict):
    """Render metrics in a single responsive row."""
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        c.metric(k, v)

# ───────────────────────────────
# Caches & DB clients
# ───────────────────────────────
@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)

@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None):
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

@st.cache_resource
def get_mongo_client(uri: str):
    return MongoClient(uri, serverSelectionTimeoutMS=2500)

@st.cache_data(ttl=30)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list, buster: int = 0):
    """
    buster: allow user to bypass cache by changing value.
    """
    db = _client[db_name]
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()

# ───────────────────────────────
# Helpers
# ───────────────────────────────
def mongo_overview(client: MongoClient, db_name: str) -> dict:
    try:
        info = client.server_info()
        db = client[db_name]
        colls = db.list_collection_names()
        stats = db.command("dbstats")
        total_docs = sum(db[c].estimated_document_count() for c in colls) if colls else 0
        return {
            "DB": db_name,
            "Collections": f"{len(colls):,}",
            "Total docs (est.)": f"{total_docs:,}",
            "Storage": f"{round(stats.get('storageSize',0)/1024/1024,1)} MB",
            "Version": info.get("version", "unknown")
        }
    except Exception as e:
        st.error(f"Mongo 连接失败：{e}")
        return {"DB": db_name, "Collections": "0", "Total docs (est.)": "0", "Storage": "0 MB", "Version": "unknown"}

def _coerce_params(raw: dict) -> dict:
    """
    Safely coerce sidebar params to expected types by name.
    This avoids accidental wrong types crashing queries.
    """
    out = {}
    int_keys = {"doctor_id", "nurse_id", "age_threshold", "days", "med_low_threshold", "reorder_threshold"}
    str_keys = {"patient_name"}

    for k, v in raw.items():
        if k in int_keys:
            try:
                out[k] = int(v)
            except Exception:
                raise ValueError(f"参数 {k} 需要整数，实际为 {v!r}")
        elif k in str_keys:
            out[k] = str(v)
        else:
            out[k] = v
    return out

def _validate_chart_spec(df: pd.DataFrame, spec: dict) -> tuple[bool, str]:
    """
    Basic validation to prevent user-supplied spec from breaking.
    """
    ctype = spec.get("type", "table")
    need = {
        "table": [],
        "line": ["x", "y"],
        "bar": ["x", "y"],
        "pie": ["names", "values"],
        "heatmap": ["rows", "cols", "values"],
        "treemap": ["path", "values"]
    }
    for col in need.get(ctype, []):
        if col not in spec:
            return False, f"图表配置缺少 '{col}' 字段"
    # columns existence check if applicable
    cols_to_check = []
    if ctype in ("line", "bar"):
        cols_to_check = [spec["x"], spec["y"]]
    elif ctype == "pie":
        cols_to_check = [spec["names"], spec["values"]]
    elif ctype == "treemap":
        cols_to_check = [spec["values"]]
    if cols_to_check:
        for c in cols_to_check:
            if isinstance(c, list):
                for cc in c:
                    if cc not in df.columns:
                        return False, f"数据缺少列: {cc}"
            else:
                if c not in df.columns:
                    return False, f"数据缺少列: {c}"
    return True, ""

def render_chart(df: pd.DataFrame, spec: dict):
    """
    Centralized renderer with validation and defensive parsing.
    """
    if df.empty:
        st.info("No rows.")
        return
    ctype = spec.get("type", "table")

    # light datetime parse on columns referenced by spec
    def _maybe_parse(col):
        if col in df.columns and df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    for key in ("x", "y", "names", "values", "rows", "cols"):
        v = spec.get(key)
        if isinstance(v, str):
            _maybe_parse(v)

    ok, msg = _validate_chart_spec(df, spec)
    if not ok:
        st.error(msg)
        st.dataframe(df, use_container_width=True)
        return

    try:
        if ctype == "table":
            st.dataframe(df, use_container_width=True)
        elif ctype == "line":
            st.plotly_chart(px.line(df, x=spec["x"], y=spec["y"]), use_container_width=True)
        elif ctype == "bar":
            st.plotly_chart(px.bar(df, x=spec["x"], y=spec["y"]), use_container_width=True)
        elif ctype == "pie":
            st.plotly_chart(px.pie(df, names=spec["names"], values=spec["values"]), use_container_width=True)
        elif ctype == "heatmap":
            pivot = pd.pivot_table(df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean")
            st.plotly_chart(px.imshow(pivot, aspect="auto", origin="upper",
                                      labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"])),
                            use_container_width=True)
        elif ctype == "treemap":
            st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    except Exception:
        st.error("图表渲染失败，已回退为表格。")
        with st.expander("错误堆栈", expanded=False):
            st.code(traceback.format_exc())
        st.dataframe(df, use_container_width=True)

# ───────────────────────────────
# Sidebar
# ───────────────────────────────
with st.sidebar:
    st.header("Connections")
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])
    mongo_uri = st.text_input("Mongo URI", CONFIG["mongo"]["uri"])
    mongo_db = st.text_input("Mongo DB name", CONFIG["mongo"]["db_name"])
    st.divider()

    st.header("Execution")
    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")
    force_refresh = st.checkbox("Force refresh (bypass cache)", value=False, key="force_refresh")
    st.divider()

    st.header("Role & Parameters")
    role = st.selectbox("User role", ["doctor","nurse","pharmacist","guardian","manager","all"], index=5)
    doctor_id = st.number_input("doctor_id", min_value=1, value=1, step=1)
    nurse_id = st.number_input("nurse_id", min_value=1, value=2, step=1)
    patient_name = st.text_input("patient_name", value="Alice")
    age_threshold = st.number_input("age_threshold", min_value=0, value=85, step=1)
    days = st.slider("last N days", 1, 90, 7)
    med_low_threshold = st.number_input("med_low_threshold", min_value=0, value=5, step=1)
    reorder_threshold = st.number_input("reorder_threshold", min_value=0, value=10, step=1)

    PARAMS_CTX_RAW = {
        "doctor_id": doctor_id,
        "nurse_id": nurse_id,
        "patient_name": patient_name,
        "age_threshold": age_threshold,
        "days": days,
        "med_low_threshold": med_low_threshold,
        "reorder_threshold": reorder_threshold,
    }

    try:
        PARAMS_CTX = _coerce_params(PARAMS_CTX_RAW)
    except Exception as e:
        st.error(str(e))
        PARAMS_CTX = {}

# ───────────────────────────────
# Postgres panel
# ───────────────────────────────
st.subheader("Postgres")
try:
    eng = get_pg_engine(pg_uri)

    with st.expander("Run Postgres query", expanded=True):
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role.lower() in t
            return {name: q for name, q in qdict.items() if ok(q.get("tags"))}

        pg_all = CONFIG["postgres"]["queries"]
        pg_q = filter_queries_by_role(pg_all, role)

        names = list(pg_q.keys()) or ["(no queries for this role)"]
        sel = st.selectbox("Choose a saved query", names, key="pg_sel")

        if sel in pg_q:
            q = pg_q[sel]
            sql = qualify(q["sql"])
            st.code(sql, language="sql")

            run = auto_run or st.button("▶ Run Postgres", key="pg_run")
            if run:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX.get(k) for k in wanted} if wanted else {}
                if None in params.values():
                    st.error("缺少必要参数，请检查侧边栏。")
                else:
                    df = run_pg_query(eng, sql, params=params)
                    render_chart(df, q["chart"])
        else:
            st.info("No Postgres queries tagged for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")

# ───────────────────────────────
# Mongo panel
# ───────────────────────────────
if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB")
    try:
        mongo_client = get_mongo_client(mongo_uri)
        metric_row(mongo_overview(mongo_client, mongo_db))

        with st.expander("Run Mongo aggregation", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Choose a saved aggregation", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]
            st.write(f"**Collection:** `{q['collection']}`")
            st.code(str(q["aggregate"]), language="python")
            runm = auto_run or st.button("▶ Run Mongo", key="mongo_run")
            if runm:
                buster = int(dt.datetime.utcnow().timestamp()) if force_refresh else 0
                dfm = run_mongo_aggregate(
                    mongo_client, mongo_db, q["collection"], q["aggregate"], buster=buster
                )
                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")
