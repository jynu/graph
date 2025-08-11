#!/usr/bin/env python3
"""
DuckDB Knowledge Graph Exporter (Fixed)
- Robust to missing tables/columns
- Proper PyVis generation with fallback to manual HTML
- Separate basic (table) HTML overview
- Cleans up duplicate/garbled code from previous version
"""

import json
import html
import os
from typing import Dict, List, Any

# Optional: PyVis for interactive graph
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except Exception:
    PYVIS_AVAILABLE = False

import duckdb


class DuckDBGraphExporter:
    def __init__(self, db_path: str = "knowledge_graph_v3.duckdb"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DuckDB file not found: {db_path}")
        # read_only avoids accidental writes
        self.conn = duckdb.connect(db_path, read_only=True)
        self.db_path = db_path

    # --------------------------- helpers ---------------------------
    def _safe_query(self, sql: str, one: bool = False, default=None):
        try:
            res = self.conn.execute(sql)
            return res.fetchone()[0] if one else res.fetchall()
        except Exception:
            return default

    def _debug_graph_structure(self):
        print("🔍 Debugging DuckDB graph structure...")
        tables_info = self._safe_query(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """,
            default=[],
        )
        if tables_info is not None:
            print("📊 Available tables:")
            for (t,) in tables_info:
                print("  -", t)

        for tname in ("tables", "columns", "relationships"):
            cnt = self._safe_query(f"SELECT COUNT(*) FROM {tname}", one=True, default=None)
            if cnt is None:
                print(f"   📋 {tname.capitalize()}: table not found")
            else:
                print(f"   📋 {tname.capitalize()}: {cnt}")

        rel_types = self._safe_query(
            """
            SELECT relationship_type, COUNT(*) AS cnt
            FROM relationships
            GROUP BY relationship_type
            ORDER BY cnt DESC
            """,
            default=[],
        )
        if rel_types:
            print("🔗 Relationship types:")
            for t, c in rel_types:
                print(f"   {t}: {c}")

        table_types = self._safe_query(
            """
            SELECT table_type, COUNT(*) AS cnt
            FROM tables
            GROUP BY table_type
            ORDER BY cnt DESC
            """,
            default=[],
        )
        if table_types:
            print("📋 Table types:")
            for t, c in table_types:
                print(f"   {t}: {c}")

    # --------------------------- data access ---------------------------
    def _fetch_tables(self):
        tables = self._safe_query(
            """
            SELECT name,
                   COALESCE(description, '') AS description,
                   COALESCE(aliases, '[]') AS aliases,
                   COALESCE(table_type, 'unknown') AS table_type
            FROM tables
            ORDER BY name
            """,
            default=[],
        )
        return tables or []

    def _fetch_relationships(self):
        rels = self._safe_query(
            """
            SELECT from_table, to_table,
                   COALESCE(from_column, '') AS from_column,
                   COALESCE(to_column, '') AS to_column,
                   COALESCE(relationship_type, 'UNKNOWN') AS relationship_type,
                   COALESCE(confidence, 0.0) AS confidence
            FROM relationships
            """,
            default=[],
        )
        return rels or []

    def _fetch_columns(self):
        cols = self._safe_query(
            """
            SELECT c.table_name, c.name AS column_name, c.full_name,
                   COALESCE(c.description, '') AS description,
                   COALESCE(c.data_type, 'unknown') AS data_type,
                   COALESCE(c.column_category, 'unknown') AS column_category
            FROM columns c
            ORDER BY c.table_name, c.name
            """,
            default=[],
        )
        return cols or []

    def _fetch_embedding_meta(self):
        try:
            row = self.conn.execute(
                "SELECT provider, dimensions, model_version FROM embedding_metadata LIMIT 1"
            ).fetchone()
            return row if row else None
        except Exception:
            return None

    # --------------------------- exports ---------------------------
    def export_to_pyvis(self, output_file: str = "knowledge_graph_fixed.html"):
        """
        Build interactive graph. If PyVis is installed, generate via pyvis, else write manual HTML.
        """
        print("🎨 Creating interactive HTML from DuckDB...")
        self._debug_graph_structure()

        tables = self._fetch_tables()
        relationships = self._fetch_relationships()
        columns = self._fetch_columns()

        # Prepare nodes/edges irrespective of backend (pyvis/manual)
        nodes_data: List[Dict[str, Any]] = []
        edges_data: List[Dict[str, Any]] = []

        # Table nodes
        for name, description, aliases_json, table_type in tables:
            try:
                aliases = json.loads(aliases_json) if aliases_json else []
                if isinstance(aliases, str):
                    aliases = [aliases]
            except Exception:
                aliases = []

            short_name = name.split(".")[-1]
            description_text = (description or "")[:200]
            tooltip = f"<b>{html.escape(name)}</b><br/>{html.escape(description_text)}"
            if aliases:
                tooltip += f"<br/><b>Aliases:</b> {html.escape(', '.join(map(str, aliases)))}"
            tooltip += f"<br/><b>Type:</b> {html.escape(table_type)}"

            # Color by table type
            up = name.upper()
            if table_type == "fact" or "FACT" in up:
                color = "#ff6b6b"
            elif table_type == "dimension" or "DIM" in up:
                color = "#4ecdc4"
            elif table_type == "reference" or "REF" in up:
                color = "#45b7d1"
            elif table_type == "market_data" or "MARKET" in up:
                color = "#f39c12"
            else:
                color = "#96ceb4"

            nodes_data.append(
                {
                    "id": name,
                    "label": short_name,
                    "title": tooltip,
                    "color": color,
                    "size": 25,
                    "shape": "box",
                    "font": {"color": "white"},
                }
            )

        # Column nodes (limit per table with simple prioritization)
        table_columns: Dict[str, List[Dict[str, str]]] = {}
        for table_name, column_name, full_name, description, data_type, column_category in columns:
            table_columns.setdefault(table_name, []).append(
                {
                    "name": column_name,
                    "full_name": full_name,
                    "description": description,
                    "data_type": data_type,
                    "category": column_category,
                }
            )

        max_per_table = 8
        for tname, cols in table_columns.items():
            cols_sorted = sorted(
                cols,
                key=lambda x: (
                    0 if x["category"] in {"id", "key"} else 1,
                    0 if "id" in x["name"].lower() else 1,
                    0 if any(k in x["name"].lower() for k in ("key", "sk")) else 1,
                    0 if "code" in x["name"].lower() else 1,
                    x["name"],
                ),
            )[:max_per_table]

            for c in cols_sorted:
                col_id = f"{tname}.{c['name']}"
                tip = f"<b>{html.escape(c['name'])}</b><br/>Table: {html.escape(tname.split('.')[-1])}"
                tip += f"<br/>Type: {html.escape(c['data_type'])}"
                tip += f"<br/>Category: {html.escape(c['category'])}"
                if c["description"]:
                    tip += f"<br/>Description: {html.escape(c['description'][:150])}"

                cname = c["name"].lower()
                cat = c["category"]
                if cat == "id" or "id" in cname:
                    col_color = "#f39c12"
                elif cat == "key" or "key" in cname:
                    col_color = "#e67e22"
                elif cat == "code" or "code" in cname:
                    col_color = "#9b59b6"
                elif "date" in cname or "time" in cname:
                    col_color = "#1abc9c"
                else:
                    col_color = "#34495e"

                nodes_data.append(
                    {
                        "id": col_id,
                        "label": c["name"],
                        "title": tip,
                        "color": col_color,
                        "size": 12,
                        "shape": "dot",
                        "font": {"color": "white"},
                    }
                )
                edges_data.append(
                    {
                        "from": tname,
                        "to": col_id,
                        "color": "#95a5a6",
                        "width": 1,
                        "title": "HAS_COLUMN",
                    }
                )

        # Relationship edges
        for from_table, to_table, from_col, to_col, rel_type, confidence in relationships:
            if from_table == to_table:
                continue
            edge_label = f"{from_col} → {to_col}" if (from_col and to_col) else ""
            # coloring
            if rel_type == "ERD_DEFINED_GEMINI":
                edge_color, edge_width = "#27ae60", 4
            elif rel_type == "BUSINESS_RULE":
                edge_color, edge_width = "#e74c3c", 5
            elif rel_type == "SURROGATE_KEY":
                edge_color, edge_width = "#9b59b6", 4
            elif rel_type == "EMBEDDING_SIMILARITY":
                edge_color, edge_width = "#3498db", 2
            elif (confidence or 0) > 0.8:
                edge_color, edge_width = "#27ae60", 4
            else:
                edge_color, edge_width = "#95a5a6", 2

            title = f"<b>{html.escape(rel_type)}</b><br/>{html.escape(from_table.split('.')[-1])} → {html.escape(to_table.split('.')[-1])}"
            if edge_label:
                title += f"<br/>Join: {html.escape(edge_label)}"
            if confidence:
                try:
                    title += f"<br/>Confidence: {float(confidence):.2f}"
                except Exception:
                    pass

            edges_data.append(
                {
                    "from": from_table,
                    "to": to_table,
                    "label": edge_label,
                    "title": title,
                    "color": edge_color,
                    "width": edge_width,
                }
            )

        # If PyVis present, build with Network; else manual html
        if PYVIS_AVAILABLE:
            try:
                net = Network(
                    height="800px",
                    width="100%",
                    bgcolor="#1e1e1e",
                    font_color="white",
                    select_menu=True,
                    filter_menu=True,
                )
                # add nodes/edges
                for n in nodes_data:
                    nid = n["id"]
                    net.add_node(
                        nid,
                        label=n["label"],
                        title=n["title"],
                        color=n["color"],
                        size=n["size"],
                        shape=n["shape"],
                    )
                for e in edges_data:
                    net.add_edge(
                        e["from"],
                        e["to"],
                        label=e.get("label", ""),
                        title=e.get("title", ""),
                        color=e.get("color", "#95a5a6"),
                        width=e.get("width", 1),
                    )
                # physics/options
                net.set_options(
                    """
                    {
                      "physics": {
                        "enabled": true,
                        "stabilization": {"iterations": 100},
                        "barnesHut": {
                          "gravitationalConstant": -8000,
                          "centralGravity": 0.3,
                          "springLength": 95,
                          "springConstant": 0.04,
                          "damping": 0.09
                        }
                      },
                      "nodes": {
                        "font": {"size": 14, "color": "white"},
                        "borderWidth": 2,
                        "shadow": true
                      },
                      "edges": {
                        "font": {"size": 12, "color": "white"},
                        "smooth": {"type": "continuous"},
                        "shadow": true
                      },
                      "interaction": {
                        "hover": true,
                        "tooltipDelay": 100,
                        "selectConnectedEdges": false
                      }
                    }"""
                )
                net.save_graph(output_file)
                # Validate
                if os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    if "vis.Network" in content and "vis.DataSet" in content:
                        print(f"✅ Interactive graph saved to: {output_file}")
                        return output_file
                    else:
                        print("⚠️ PyVis output did not contain expected content. Falling back to manual HTML.")
            except Exception as e:
                print(f"⚠️ PyVis generation failed: {e}. Falling back to manual HTML.")

        # manual write
        return self._write_manual_html(nodes_data, edges_data, tables, output_file)

    def _write_manual_html(self, nodes_data, edges_data, tables, output_file):
        nodes_json = json.dumps(nodes_data)
        edges_json = json.dumps(edges_data)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Knowledge Graph - Interactive View</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" />
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
    #mynetwork {{ width: 100%; height: 800px; background-color: #1e1e1e; border: 1px solid lightgray; }}
    .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
    h1 {{ color: #333; text-align: center; }}
    .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
    .stat {{ text-align: center; padding: 15px; background: #4CAF50; color: white; border-radius: 5px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 Interactive Knowledge Graph</h1>
    <div class="stats">
      <div class="stat"><h3>{len(tables)}</h3><p>Tables</p></div>
      <div class="stat"><h3>{max(0, len(nodes_data)-len(tables))}</h3><p>Columns</p></div>
      <div class="stat"><h3>{len(edges_data)}</h3><p>Relationships</p></div>
    </div>
    <div id="mynetwork"></div>
  </div>
  <script type="text/javascript">
    var nodes = new vis.DataSet({nodes_json});
    var edges = new vis.DataSet({edges_json});
    var container = document.getElementById('mynetwork');
    var data = {{ nodes: nodes, edges: edges }};
    var options = {{
      physics: {{
        enabled: true,
        stabilization: {{iterations: 100}},
        barnesHut: {{
          gravitationalConstant: -8000,
          centralGravity: 0.3,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.09
        }}
      }},
      nodes: {{ font: {{size: 14, color: "white"}}, borderWidth: 2, shadow: true }},
      edges: {{ font: {{size: 12, color: "white"}}, smooth: {{type: "continuous"}}, shadow: true }},
      interaction: {{ hover: true, tooltipDelay: 100 }}
    }};
    var network = new vis.Network(container, data, options);
    console.log('Knowledge graph loaded with', nodes.length, 'nodes and', edges.length, 'edges');
  </script>
</body>
</html>
"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ Manual interactive HTML saved to: {output_file}")
        return output_file

    def export_to_basic_html(self, output_file: str = "knowledge_graph_table.html"):
        """Non-graph HTML: overview tables for tables/relationships/column sample"""
        tables = self._fetch_tables()
        relationships = self._fetch_relationships()
        columns = self._safe_query(
            """
            SELECT c.table_name, c.name AS column_name, c.full_name,
                   COALESCE(c.description, '') AS description,
                   COALESCE(c.data_type, 'unknown') AS data_type,
                   COALESCE(c.column_category, 'unknown') AS column_category
            FROM columns c
            ORDER BY c.table_name, c.name
            LIMIT 500
            """,
            default=[],
        ) or []

        emb = self._fetch_embedding_meta()

        def esc(x): return html.escape(str(x)) if x is not None else ""

        html_parts = []
        html_parts.append("""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
.container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }
.table { border-collapse: collapse; width: 100%; margin: 20px 0; }
.table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
.table th { background: #f2f2f2; }
.stat { text-align:center; padding: 10px; background: #4CAF50; color: white; border-radius: 5px; }
.flex { display:flex; gap: 10px; }
.badge { display:inline-block; padding: 2px 6px; border-radius: 4px; background:#eee; }
</style></head><body><div class="container">
<h1>DuckDB Knowledge Graph Overview</h1>
""")
        html_parts.append(f"<p><b>Database:</b> {esc(self.db_path)} "
                          f" | <b>Size:</b> {os.path.getsize(self.db_path)/(1024*1024):.2f} MB</p>")
        if emb:
            html_parts.append(
                f"<p class='badge'>Embedding: {esc(emb[0])} / dim={esc(emb[1])} / model={esc(emb[2])}</p>"
            )

        html_parts.append(
            f"""<div class="flex">
<div class="stat"><h3>{len(tables)}</h3>Tables</div>
<div class="stat"><h3>{len(columns)}</h3>Columns (sample)</div>
<div class="stat"><h3>{len(relationships)}</h3>Relationships</div>
</div>"""
        )

        # tables
        html_parts.append("<h2>Tables</h2><table class='table'><tr><th>Name</th><th>Type</th><th>Description</th></tr>")
        for name, description, aliases_json, table_type in tables:
            short = name.split(".")[-1]
            html_parts.append(f"<tr><td><code>{esc(short)}</code></td><td>{esc(table_type)}</td><td>{esc(description[:300])}</td></tr>")
        html_parts.append("</table>")

        # relationships
        html_parts.append("<h2>Relationships</h2><table class='table'><tr><th>From</th><th>To</th><th>Join</th><th>Type</th><th>Confidence</th></tr>")
        for from_table, to_table, from_col, to_col, rel_type, confidence in relationships:
            join = f"{from_col} → {to_col}" if (from_col and to_col) else ""
            html_parts.append(f"<tr><td><code>{esc(from_table.split('.')[-1])}</code></td>"
                              f"<td><code>{esc(to_table.split('.')[-1])}</code></td>"
                              f"<td>{esc(join)}</td><td>{esc(rel_type)}</td><td>{float(confidence):.2f}</td></tr>")
        html_parts.append("</table>")

        # columns (sample)
        html_parts.append("<h2>Columns (sample)</h2><table class='table'><tr><th>Table</th><th>Column</th><th>Category</th><th>Type</th><th>Description</th></tr>")
        for table_name, column_name, full_name, description, data_type, column_category in columns[:200]:
            html_parts.append(f"<tr><td><code>{esc(table_name.split('.')[-1])}</code></td>"
                              f"<td>{esc(column_name)}</td><td>{esc(column_category)}</td>"
                              f"<td>{esc(data_type)}</td><td>{esc(description[:200])}</td></tr>")
        html_parts.append("</table></div></body></html>")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(html_parts))

        print(f"✅ Basic overview HTML saved to: {output_file}")
        return output_file

    # --------------------------- stats ---------------------------
    def get_graph_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        stats["table_count"] = self._safe_query("SELECT COUNT(*) FROM tables", one=True, default=0) or 0
        stats["column_count"] = self._safe_query("SELECT COUNT(*) FROM columns", one=True, default=0) or 0
        stats["relationship_count"] = self._safe_query("SELECT COUNT(*) FROM relationships", one=True, default=0) or 0
        stats["value_count"] = self._safe_query("SELECT COUNT(*) FROM values", one=True, default=0) or 0

        stats["table_types"] = dict(
            self._safe_query("SELECT table_type, COUNT(*) FROM tables GROUP BY table_type ORDER BY 2 DESC", default=[])
            or []
        )
        stats["relationship_types"] = dict(
            self._safe_query(
                "SELECT relationship_type, COUNT(*) FROM relationships GROUP BY relationship_type ORDER BY 2 DESC",
                default=[],
            )
            or []
        )
        stats["column_categories"] = dict(
            self._safe_query(
                "SELECT column_category, COUNT(*) FROM columns GROUP BY column_category ORDER BY 2 DESC", default=[]
            )
            or []
        )

        emb = self._fetch_embedding_meta()
        if emb:
            stats["embedding_provider"], stats["embedding_dimensions"], stats["embedding_model"] = emb
        else:
            stats["embedding_provider"] = "Unknown"
        return stats

    def close(self):
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    print("🚀 DuckDB Knowledge Graph Exporter (Fixed)")
    print("=" * 60)
    # locate db
    db_path = "knowledge_graph_v3.duckdb"
    if not os.path.exists(db_path):
        candidates = [
            "knowledge_graph.duckdb",
            "knowledge_graph_v2.duckdb",
            "kg_v3.duckdb",
            "kg_builder_v2.duckdb",
        ]
        print(f"❌ Default database '{db_path}' not found.")
        print("🔍 Searching for alternative database files...")
        for p in candidates:
            if os.path.exists(p):
                db_path = p
                print(f"✅ Found database: {p}")
                break
        else:
            print("❌ No DuckDB database found next to the script.")
            print("   Please place your DuckDB file (e.g., knowledge_graph_v3.duckdb) in the same folder and rerun.")
            return

    print(f"✅ Using database: {db_path}")
    exporter = DuckDBGraphExporter(db_path)

    stats = exporter.get_graph_statistics()
    print("📊 Stats:", stats)

    print("1️⃣ Building interactive graph...")
    html_graph = exporter.export_to_pyvis("knowledge_graph_fixed.html")

    print("2️⃣ Building table overview...")
    html_table = exporter.export_to_basic_html("knowledge_graph_table.html")

    exporter.close()
    print("=" * 60)
    print("✅ Done.")
    print("Output files:")
    print(" -", html_graph)
    print(" -", html_table)


if __name__ == "__main__":
    main()
