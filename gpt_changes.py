Backend — text_to_sql_backend (3).py
A) Add two helpers inside AdvancedGraphTraversalRetriever
Add _path_diversity_score(...) (anywhere among the other private helpers in the class):

python
Copy
Edit
def _path_diversity_score(self, table_name: str, seed_tables: List[str]) -> float:
    """Fraction of distinct seed tables that can reach `table_name` within max_hops."""
    if not seed_tables or not self.graph_structure.nodes():
        return 0.0
    reachable_from = 0
    for seed in seed_tables:
        if seed == table_name:
            reachable_from += 1
            continue
        # bounded BFS up to self.max_hops
        try:
            lengths = nx.single_source_shortest_path_length(self.graph_structure, seed, cutoff=self.max_hops)
            if table_name in lengths:
                reachable_from += 1
        except Exception:
            pass
    return reachable_from / max(1, len(seed_tables))
Add _compute_similarity_components(...) (near other similarity/attention helpers):

python
Copy
Edit
def _compute_similarity_components(
    self, table_name: str, query: str, query_embedding: np.ndarray, seed_tables: List[str]
) -> Dict[str, float]:
    # embedding
    table_emb = self._get_table_embedding(table_name)
    emb_sim = self._cosine_similarity(query_embedding, table_emb) if table_emb is not None else 0.0

    # structure
    struct = self._get_table_centrality(table_name)

    # attention (your existing query–table relevance)
    attn = self._compute_query_table_relevance(table_name, query)

    # path diversity
    path_div = self._path_diversity_score(table_name, seed_tables)

    return {
        "embedding": float(emb_sim),
        "structure": float(struct),
        "attention": float(attn),
        "path_diversity": float(path_div),
    }
(You already have _get_table_embedding, _cosine_similarity, _get_table_centrality, and _compute_query_table_relevance available in this class. )

B) Update get_tables_with_details(...) to compute & log components
Function: AdvancedGraphTraversalRetriever.get_tables_with_details(...)
Where: After you have embedding_results, multihop_results, column_matches, and before enriching details. (See the existing flow.)

Add the block below right after you compute filtered_results (and possibly after your fallback selection):

python
Copy
Edit
# Pick seeds for path-diversity (top-5 by embedding attention)
seed_tables = embedding_results[:5]

# Precompute query embedding is already available as query_embedding
component_breakdown = {}

for t in filtered_results:
    comps = self._compute_similarity_components(t, query, query_embedding, seed_tables)
    component_breakdown[t] = comps

    total = (
        comps["embedding"] * self.embedding_weight +
        comps["structure"] * self.structure_weight +
        comps["attention"] * self.attention_weight +
        comps["path_diversity"] * self.path_diversity_weight
    )

    logger.info(
        f"[SIM] {t}: emb={comps['embedding']:.3f}, "
        f"struct={comps['structure']:.3f}, "
        f"attn={comps['attention']:.3f}, "
        f"path={comps['path_diversity']:.3f} | total={total:.3f}"
    )
We’ll attach this breakdown to table_details in the next step.

C) Extend _enrich_details_with_similarity_scores(...) to include components
Function: _enrich_details_with_similarity_scores(self, table_details, query, all_tables, all_scores)
Additions: copy component scores + weights into each table’s dict. (This function already adds similarity_score, rank, confidence.)

Directly after you set confidence, add:

python
Copy
Edit
# If component_breakdown was computed in caller, attach it
try:
    # This expects a local dict named component_breakdown in the caller; so we’ll pass it via a closure:
    pass
except:
    pass
Since the function signature doesn’t carry the dict yet, the simplest minimal-change pattern is:

Before calling _enrich_details_with_similarity_scores(...) in get_tables_with_details(...), inject the dict into self.attention_weights (you already use that dict). Then read it here.

In get_tables_with_details(...), after computing component_breakdown (previous step), add:

python
Copy
Edit
# Reuse existing container for per-table data
self.attention_weights.update(component_breakdown)
Then, inside _enrich_details_with_similarity_scores(...), append:

python
Copy
Edit
# Component breakdown (if available)
if hasattr(self, "attention_weights"):
    comps = self.attention_weights.get(table_name)
    if comps:
        table_details[table_name]["similarity_components"] = {
            "embedding": round(float(comps.get("embedding", 0.0)), 3),
            "structure": round(float(comps.get("structure", 0.0)), 3),
            "attention": round(float(comps.get("attention", 0.0)), 3),
            "path_diversity": round(float(comps.get("path_diversity", 0.0)), 3),
        }
        table_details[table_name]["component_weights"] = {
            "embedding_weight": self.embedding_weight,
            "structure_weight": self.structure_weight,
            "attention_weight": self.attention_weight,
            "path_diversity_weight": self.path_diversity_weight,
        }
D) Show the components in the details formatter (UI consumes this)
Function: format_table_details_for_display(table_details: Dict) -> str
Append a “Similarity” block after the basic table info/columns. (This function is already used by the frontend “Detailed Table Information” accordion.)

Right before the final formatted_text += "\n---\n", add:

python
Copy
Edit
# Similarity block
sim_score = details.get("similarity_score", None)
comps = details.get("similarity_components", None)
if sim_score is not None or comps:
    formatted_text += "**Similarity (final):** "
    formatted_text += f"{sim_score:.3f}\n" if sim_score is not None else "N/A\n"
    if comps:
        formatted_text += "**Components:**\n"
        formatted_text += f"- **Embedding:** {comps.get('embedding', 0.0):.3f}\n"
        formatted_text += f"- **Structure:** {comps.get('structure', 0.0):.3f}\n"
        formatted_text += f"- **Attention:** {comps.get('attention', 0.0):.3f}\n"
        formatted_text += f"- **Path diversity:** {comps.get('path_diversity', 0.0):.3f}\n"
Frontend — text_to_sql_frontend (7).py
E) Add compact component chips in the ranked summary
Function: create_table_summary_with_scores(tables, table_details)
(You already render Score:; we’ll add [E | S | A | P] chips inline.)

Inside the loop where each line is formed (right before the final append), insert:

python
Copy
Edit
comps = table_details.get(table_name, {}).get('similarity_components', {})
chip = ""
if comps:
    chip = f" [E {comps.get('embedding',0):.2f} | S {comps.get('structure',0):.2f} | A {comps.get('attention',0):.2f} | P {comps.get('path_diversity',0):.2f}]"
Then change the final line from:

python
Copy
Edit
summary_text += f"**#{rank}** `{display_name}` ({table_type}) *Score: {score:.3f}*{desc_text}\n"
to:

python
Copy
Edit
summary_text += f"**#{rank}** `{display_name}` ({table_type}) *Score: {score:.3f}*{chip}{desc_text}\n"
(That loop sits in the function body shown here.)

No other UI wiring changes are needed; the “Detailed Table Information” accordion will automatically display the new block because it already calls the backend formatter. (See where the frontend obtains the formatted details.)