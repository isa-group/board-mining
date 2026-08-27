<script>
  import { onMount } from 'svelte'
  import { getCardIndicators, getCardTimeline } from '../api.js'
  import { selectedList, selectedCard, selectedTransition } from '../stores.js'

  let cards = $state([])
  let filtered = $state([])
  let error = $state(null)
  let timeline = $state(null)
  let timelineLoading = $state(false)
  let sortKey = $state('bouncing')
  let sortDesc = $state(true)

  // ── Drag-resize state ───────────────────────────────────────────────────────
  let panesEl                   // DOM ref for the panes container
  let drawerPct = $state(25)    // drawer height as % of panes height (25% default)
  let isDragging = false        // plain var — only read inside event handler closures

  function startResize(e) {
    isDragging = true
    e.preventDefault()
  }

  const COLUMNS = [
    { key: 'card_id',     label: 'Card ID' },
    { key: 'bouncing',    label: 'Bounces',      numeric: true },
    { key: 'silent_moves',label: 'Silent moves', numeric: true },
    { key: 'inactive',    label: 'Inactive',     bool: true },
    { key: 'orphan',      label: 'Orphan',       bool: true },
    { key: 'overdue',     label: 'Overdue',      bool: true },
    { key: 'unassigned',  label: 'Unassigned',   bool: true },
  ]

  async function loadCards() {
    const tr = $selectedTransition
    const sl = $selectedList
    try {
      cards = await getCardIndicators(tr?.source ?? null, tr?.target ?? null, sl)
      applyFilter()
    } catch (e) {
      error = e.message
    }
  }

  // Re-fetch whenever the transition or list filter changes
  $effect(() => {
    $selectedTransition
    $selectedList
    loadCards()
  })

  // Synchronous onMount so that the returned cleanup function is called correctly.
  // async onMount returns a Promise, which Svelte cannot use as a cleanup function.
  onMount(() => {
    function handleMove(e) {
      if (!isDragging || !panesEl) return
      const rect = panesEl.getBoundingClientRect()
      const fromBottom = rect.bottom - e.clientY
      drawerPct = Math.min(75, Math.max(15, Math.round((fromBottom / rect.height) * 100)))
    }
    function handleUp() { isDragging = false }

    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)
    return () => {
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
    }
  })

  function applyFilter() {
    let rows = cards

    rows = [...rows].sort((a, b) => {
      const av = a[sortKey] ?? 0
      const bv = b[sortKey] ?? 0
      return sortDesc ? bv - av : av - bv
    })
    filtered = rows
  }

  $effect(() => { sortKey; sortDesc; applyFilter() })

  function toggleSort(key) {
    if (sortKey === key) sortDesc = !sortDesc
    else { sortKey = key; sortDesc = true }
  }

  async function openTimeline(cardId) {
    selectedCard.set(cardId)
    timelineLoading = true
    timeline = null
    try {
      timeline = await getCardTimeline(cardId)
    } finally {
      timelineLoading = false
    }
  }

  function closeTimeline() {
    selectedCard.set(null)
    timeline = null
  }

  function flag(val, key) {
    if (val == null) return '–'
    if (typeof val === 'boolean') return val ? '⚠' : '✓'
    return val
  }

  function flagClass(val, key) {
    if (typeof val !== 'boolean') return ''
    return val ? 'bad' : 'ok'
  }
</script>

<div class="cards-tab">
  <div class="section-header">
    <h2 class="section-title">Card health indicators</h2>
    <p class="hint">
      Click a row to inspect that card's move history ·
      {filtered.length} of {cards.length} cards shown
    </p>
  </div>

  {#if $selectedTransition}
    <div class="filter-chip">
      <span class="filter-label">Transition filter:</span>
      <span class="filter-value">{$selectedTransition.source} → {$selectedTransition.target}</span>
      <button class="filter-clear" onclick={() => selectedTransition.set(null)} title="Clear filter">✕</button>
    </div>
  {/if}

  {#if $selectedList}
    <div class="filter-chip">
      <span class="filter-label">List filter:</span>
      <span class="filter-value">{$selectedList}</span>
      <button class="filter-clear" onclick={() => selectedList.set(null)} title="Clear filter">✕</button>
    </div>
  {/if}

  <div class="panes" bind:this={panesEl}>
    {#if error}
      <p class="error">{error}</p>
    {:else if !cards.length}
      <p class="loading">Loading…</p>
    {:else}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              {#each COLUMNS as col}
                <th
                  class:sortable={col.numeric || col.key === 'card_id'}
                  class:sorted={sortKey === col.key}
                  onclick={() => (col.numeric || col.key === 'card_id') && toggleSort(col.key)}
                >
                  {col.label}
                  {#if sortKey === col.key}{sortDesc ? ' ↓' : ' ↑'}{/if}
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each filtered as card}
              <tr
                class:selected={$selectedCard === card.card_id}
                onclick={() => openTimeline(card.card_id)}
              >
                {#each COLUMNS as col}
                  <td class={flagClass(card[col.key], col.key)}>
                    {flag(card[col.key], col.key)}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    {#if $selectedCard}
      <div class="resizer" role="separator" onmousedown={startResize}></div>
      <div class="drawer" style="flex: 0 0 {drawerPct}%">
        <div class="drawer-header">
          <span class="drawer-title">Card · {$selectedCard}</span>
          <button class="close-btn" onclick={closeTimeline}>✕</button>
        </div>
        {#if timelineLoading}
          <p class="loading">Loading timeline…</p>
        {:else if timeline}
          <div class="timeline-wrap">
            <table class="timeline-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Event type</th>
                  <th>From list</th>
                  <th>To list</th>
                  <th>Actor</th>
                </tr>
              </thead>
              <tbody>
                {#each timeline as ev}
                  <tr>
                    <td>{ev.timestamp ?? '–'}</td>
                    <td><span class="badge">{ev.card_event_type ?? ev.raw_event_type ?? '–'}</span></td>
                    <td>{ev.source_list_name ?? '–'}</td>
                    <td>{ev.target_list_name ?? '–'}</td>
                    <td>{ev.actor_id ?? '–'}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* ── Top-level layout ─────────────────────────────────────────────────────── */
  .cards-tab {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    gap: 12px;
  }

  .section-header { flex: 0 0 auto; display: flex; flex-direction: column; gap: 4px; }

  .filter-chip {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(108, 142, 245, 0.12);
    border: 1px solid var(--accent-dim);
    border-radius: 20px;
    padding: 4px 10px 4px 12px;
    font-size: 12px;
    align-self: flex-start;
  }

  .filter-label { color: var(--text-muted); }
  .filter-value { color: var(--accent); font-weight: 500; }

  .filter-clear {
    background: none;
    color: var(--text-muted);
    font-size: 11px;
    padding: 1px 4px;
    border-radius: 50%;
    line-height: 1;
    transition: background 0.15s, color 0.15s;
  }
  .filter-clear:hover { background: var(--accent-dim); color: var(--text); }
  .section-title { font-size: 15px; font-weight: 600; }
  .hint { font-size: 12px; color: var(--text-muted); }

  /* ── Two-pane container ───────────────────────────────────────────────────── */
  .panes {
    flex: 1 1 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-height: 0;
    gap: 0;
  }

  .table-wrap {
    flex: 1 1 0;
    min-height: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: auto;
  }

  table { width: 100%; border-collapse: collapse; }

  th {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    user-select: none;
  }

  th.sortable { cursor: pointer; }
  th.sortable:hover { color: var(--text); }
  th.sorted { color: var(--accent); }

  td {
    padding: 9px 14px;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  td.bad { color: var(--danger); font-weight: 600; }
  td.ok  { color: var(--success); }

  tbody tr { cursor: pointer; transition: background 0.1s; }
  tbody tr:hover { background: var(--surface-2); }
  tbody tr.selected { background: rgba(108, 142, 245, 0.1); }

  /* ── Resizer handle ──────────────────────────────────────────────────────── */
  .resizer {
    flex: 0 0 8px;
    margin: 3px 0;
    border-radius: 4px;
    background: var(--border);
    cursor: ns-resize;
    transition: background 0.15s;
  }
  .resizer:hover { background: var(--accent); }

  /* ── Detail drawer ───────────────────────────────────────────────────────── */
  .drawer {
    /* flex-basis set inline via drawerPct; JS already clamps to [15%, 75%] */
    min-height: 100px;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }

  .drawer-header {
    flex: 0 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
  }

  .drawer-title { font-size: 13px; font-weight: 600; }

  .close-btn {
    background: none;
    color: var(--text-muted);
    font-size: 14px;
    padding: 2px 6px;
    border-radius: 4px;
  }

  .close-btn:hover { background: var(--surface-2); color: var(--text); }

  /* timeline fills the remaining drawer height and scrolls internally */
  .timeline-wrap { flex: 1; overflow: auto; }
  .timeline-table { width: 100%; }
  .timeline-table th { font-size: 11px; }
  .timeline-table td { font-size: 12px; }

  .badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 10px;
    background: var(--accent-dim);
    color: var(--accent);
    font-size: 11px;
    font-weight: 500;
  }

  .loading, .error { color: var(--text-muted); padding: 24px 0; }
  .error { color: var(--danger); }
</style>
