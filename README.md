# bomi: Board Mining Library

`bomi` is a Python library for analyzing event logs from board-based collaborative work management tools such as Trello and Jira. It provides:

- **Board discovery** – Identify board structure, list roles, and flow patterns
- **Board evolution** – Track changes and redesigns over time
- **Health metrics** – Assess board quality, completion rates, and collaboration discipline
- **Visualization** – Interactive dashboards and graph visualizations
- **Process mining export** – Convert board logs to process mining formats

## Installation

Install the package locally from this repository:

```bash
pip install -e .
```

Install the optional process-mining dependency when using `to_event_log`:

```bash
pip install -e ".[process]"
```

Install development dependencies for testing:

```bash
pip install -e ".[dev]"
pytest
```

## Quickstart

```python
import bomi

df = bomi.load_board("<trello-board-id>")
info = bomi.log_info(df)
redesigns = bomi.detect_redesign(df, threshold=20, threshold_l_events=2)
metrics = bomi.static_metrics(df, redesigns=redesigns)
discovery = bomi.board_discovery(df)
```

The legacy analysis functions currently expect a pandas `DataFrame` with Trello-style event-log columns. New connectors and analysis modules should use the canonical schema described below.

## Core Functionality

### Importers and Exporters

Use the IO helpers to load board event logs into the canonical schema:

```python
canonical = bomi.load_trello_board("<trello-board-id>")
canonical = bomi.read_trello_json("actions.json")
canonical = bomi.read_board_csv("actions.csv")
canonical = bomi.from_dataframe(df)
```

Write canonical logs back to CSV:

```python
bomi.write_board_csv(canonical, "canonical-actions.csv")
```

Convert canonical logs back to the flattened Trello column names expected by the legacy analysis functions:

```python
legacy = bomi.to_trello_log(canonical)
bomi.enrich_log(legacy)
metrics = bomi.static_metrics(legacy)
```

Export a canonical log to a process-mining-compatible pandas DataFrame:

```python
process_df = bomi.to_process_dataframe(canonical)
```

### Canonical Board Event-Log Schema

The canonical schema provides a tool-independent representation of board event logs. Trello columns can be converted with:

```python
canonical = bomi.to_canonical_log(df)
report = bomi.validate_canonical_log(canonical)
```

**Required columns:**

| Column | Description |
| --- | --- |
| `event_id` | Unique event identifier |
| `event_type` | Normalized event type |
| `timestamp` | Event timestamp as a pandas-compatible datetime |

**Recommended entity columns:**

| Column | Description |
| --- | --- |
| `actor_id` | User or member that produced the event |
| `board_id` | Board identifier |
| `board_name` | Board name when available |
| `card_id` | Card identifier |
| `card_name` | Card name when available |
| `list_id` | Current list identifier |
| `list_name` | Current list name |

**State and transition columns:**

| Column | Description |
| --- | --- |
| `card_closed` | Whether the card is closed after the event |
| `card_due` | Card due date when available |
| `list_closed` | Whether the list is closed after the event |
| `source_list_id` | Origin list identifier for card movements |
| `source_list_name` | Origin list name for card movements |
| `target_list_id` | Destination list identifier for card movements |
| `target_list_name` | Destination list name for card movements |
| `old_name` | Previous name for rename events |

**Provenance columns:**

| Column | Description |
| --- | --- |
| `raw_event_type` | Original event type from the source system |
| `source_system` | Source tool (e.g., `trello`, `jira`) |

### Board Discovery

Automatically discover the structure and design of your board:

```python
# Discover board structure with default parameters
board = bomi.board_discovery(df)
# Returns: lists, card flow (connected components), semantic precedence, roles

# Access discovery results
print(board.lists)                    # All lists on the board
print(board.card_flow)                # Connected components (workflows)
print(board.semantic_precedence)      # Dominant flow patterns (source → target)
print(board.card_create_lists)        # Lists where cards originate
print(board.card_close_lists)         # Lists where cards complete
print(board.card_use_lists)           # Lists where cards are updated
```

**Threshold parameters** control structure discovery (as percentages, 0-100):

```python
board = bomi.board_discovery(
    df,
    cf_threshold=0,      # Card flow connectivity (0 = all edges)
    cc_threshold=0,      # Card creation list threshold
    cx_threshold=0,      # Card close list threshold
    cu_threshold=0,      # Card use list threshold
    sp_threshold=0,      # Semantic precedence threshold
)
```

Lower thresholds include more flows/roles; higher thresholds keep only the dominant patterns.

### Board Evolution

Track how your board structure changes over time:

```python
# Detect periods of board redesign
redesigns = bomi.detect_redesign(df, threshold=20, threshold_l_events=2)
# Returns: list of redesign periods with (start, end) timestamps

# Analyze board changes between redesigns
board_over_time = bomi.board_evolution(df, bins="7D")
# Returns: DataFrame with event counts by category over time
```

**Redesign detection** identifies when the board structure changed significantly:

```python
# Customize redesign detection sensitivity
redesigns = bomi.detect_redesign(
    df,
    threshold=20,              # Minimum events to trigger redesign
    threshold_l_events=2,      # Minimum list-level events
)

# Access redesign details
for start, end in redesigns:
    print(f"Redesign period: {start} to {end}")
```

**Evolution analysis** shows how board activity changes:

```python
# Board evolution with different time bins
evolution_weekly = bomi.board_evolution(df, bins="7D")
evolution_monthly = bomi.board_evolution(df, bins="30D")

# Returns DataFrame indexed by time with columns:
# - card_create, card_move, card_close, card_delete
# - list_create, list_rename, list_move, etc.
```

### Board Health and Quality Indicators

`bomi.health` provides board health metrics organized in four layers.

#### Completion Method

All indicators that distinguish active from completed cards accept a `method` parameter:

| Method | Meaning |
| --- | --- |
| `"archived"` (default) | Card ever had `card_closed == True` |
| `"sink_list"` | Card's last list is in `sink_lists` |
| `"deleted"` | Card had a `card_delete` event |

Methods can be combined as a list, e.g., `method=["archived", "sink_list"]`.

#### Per-card Indicators

```python
age      = bomi.card_age(df, reference_date=None, method="archived", sink_lists=None)
inactive = bomi.inactive_cards(df, window=pd.Timedelta("30D"), reference_date=None)
orphans  = bomi.orphan_cards(df)
overdue  = bomi.overdue_cards(df, reference_date=None)
bounces  = bomi.bouncing_cards(df)
silent   = bomi.silent_moves(df)
unassign = bomi.unassigned_cards(df, reference_date=None)   # None if no assignment events
```

#### Per-list Indicators

```python
stagnant = bomi.stagnant_lists(df, window=pd.Timedelta("30D"), reference_date=None)
dead     = bomi.dead_lists(df, window=pd.Timedelta("30D"), reference_date=None)
```

#### Board-level Metrics

```python
# prescribed_flow is a list of (source_list_name, target_list_name) pairs;
# when None the dominant flow is inferred from the data
fc   = bomi.flow_conformance(df, prescribed_flow=None, infer_threshold=0.05)
comp = bomi.completion_rate(df)
aband = bomi.abandonment_rate(df)
```

#### Dimension Scores

```python
# Five aggregated scores in [0, 1]
dims = bomi.health_dimensions(df)
# Keys: flow_discipline, collaboration_discipline, completion_discipline,
#       structural_stability, board_vitality

# Complete report (scalar rates + counts + dim_* scores)
health = bomi.board_health(df, sink_lists=["Done"], method=["archived", "sink_list"])
```

#### Temporal Evolution

Track health metrics over time using a sliding window:

```python
# Compute all health indicators at regular intervals
evolution = bomi.health_evolution(
    df,
    window=pd.Timedelta("30D"),   # lookback window for time-based indicators
    step=pd.Timedelta("7D"),      # interval between reference dates
)
# Returns a DataFrame indexed by timestamp; one column per board_health() key

# Track only specific indicators
evolution = bomi.health_evolution(
    df,
    window=pd.Timedelta("30D"),
    step=pd.Timedelta("7D"),
    indicators=["dim_flow_discipline", "dim_board_vitality", "completion_rate"],
    prescribed_flow=[("Backlog", "InProgress"), ("InProgress", "Done")],
)
```

Each row is a snapshot of the board's health at that point in time, using all events up to (and including) the reference date. Passing an explicit `prescribed_flow` is recommended when tracking `flow_conformance` over time; otherwise the dominant flow is re-inferred from each slice independently.

## Examples and Datasets

### Board Event Logs Dataset

We have collected a [dataset of 616 board event logs](https://drive.google.com/file/d/1D5rybwE4dx1vMQyrOHki1GuQXZdrHbvw/view?usp=sharing) from public Trello boards. You can download the full dataset or replicate the download process:

1. Use `index.js` with Node.js to download all JSON files
2. Use `notebook.ipynb` – a Jupyter Notebook that transforms downloaded JSON files into CSV format

### Empirical Analysis

From the 616 board event logs, we filtered to keep logs of boards that:
- Represent their entire life (i.e., the log starts with a board creation event)
- Have over 2,000 events and over 12 weeks of use

This resulted in 63 logs for detailed analysis. The procedure is detailed in:

1. `analysis.ipynb` – Jupyter Notebook showing filtering and metric generation
2. `details.xlsx` – Metrics computed in the notebook with manual design pattern categorization

**Key findings:** Board use often lacks discipline, with metric values averaging under 50%, indicating deficient use patterns. Automated support for detecting misaligned cards and monitoring board use would help users maintain better board discipline.

### Use Cases

See the following notebooks for detailed analysis of three representative boards:

- `analysis_Oeagag_Trello.ipynb`
- `analysis_Wooting_roadmap.ipynb`
- `analysis_Zwar57_Sheets.ipynb`

These demonstrate how board mining techniques can reveal insights into board structure, use patterns, and evolution.
