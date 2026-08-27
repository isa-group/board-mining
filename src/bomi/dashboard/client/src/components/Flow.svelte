<script>
  import { getTransitionMatrix, getListEvolution, getBoardDesign } from '../api.js'
  import { selectedTransition, redesignPeriods, flowShowPerRedesign, flowSelectedRedesign } from '../stores.js'
  import * as d3 from 'd3'
  import FlowGraph from './FlowGraph.svelte'
  import BoardDesignEditor from './BoardDesignEditor.svelte'

  let matrixData = $state(null)
  let boardDesignData = $state(null)
  let error      = $state(null)
  let svgEl      = $state(null)
  let viewMode   = $state('matrix')   // 'matrix' | 'graph' | 'design'
  let sortBy     = $state('net_flow') // 'net_flow' | 'topological'
  let movementsThreshold = $state(0)  // For both cardflow and semantic precedence
  let ccThreshold = $state(0)
  let cxThreshold = $state(0)
  let cuThreshold = $state(0)
  let interRedesignPeriods = $state([])  // periods between/around redesigns
  let globalStartDate = $state(null)
  let globalEndDate = $state(null)
  let boardDesignLoading = $state(false)

  // Bind to stores so values persist across tabs
  let showPerRedesign = $derived($flowShowPerRedesign)
  let selectedRedesign = $derived($flowSelectedRedesign)

  $effect(() => {
    if (viewMode === 'matrix' && matrixData && svgEl) drawMatrix()
  })

  // Re-draw when active cell changes (to update highlight ring)
  $effect(() => {
    $selectedTransition
    if (viewMode === 'matrix' && matrixData && svgEl) drawMatrix()
  })

  // Get global date range on mount
  $effect(() => {
    getListEvolution()
      .then(lists => {
        if (lists.length > 0) {
          const allDates = lists.flatMap(l => [l.begin_date, l.last_date]).filter(Boolean)
          globalStartDate = new Date(Math.min(...allDates.map(d => new Date(d))))
          globalEndDate = new Date(Math.max(...allDates.map(d => new Date(d))))
        }
      })
      .catch(e => { error = e.message })
  })

  // Compute inter-redesign periods from cached redesigns (including first and last)
  $effect(() => {
    const redesigns = $redesignPeriods
    if (!redesigns || redesigns.length === 0 || !globalStartDate || !globalEndDate) {
      interRedesignPeriods = []
      return
    }

    // Sort by min timestamp
    const sorted = [...redesigns].sort((a, b) => new Date(a.min) - new Date(b.min))

    const periods = []

    // Period from beginning to first redesign
    if (globalStartDate < new Date(sorted[0].min)) {
      periods.push({
        start: globalStartDate.toISOString(),
        end: sorted[0].min,
        label: `Beginning to Redesign 1`,
      })
    }

    // Periods between redesigns
    for (let i = 0; i < sorted.length - 1; i++) {
      periods.push({
        start: sorted[i].max,
        end: sorted[i + 1].min,
        label: `Between Redesign ${i + 1} & ${i + 2}`,
      })
    }

    // Period from last redesign to end
    if (new Date(sorted[sorted.length - 1].max) < globalEndDate) {
      periods.push({
        start: sorted[sorted.length - 1].max,
        end: globalEndDate.toISOString(),
        label: `Redesign ${sorted.length} to End`,
      })
    }

    interRedesignPeriods = periods
  })

  // Load matrix data based on per-redesign mode and sorting
  $effect(() => {
    if ($flowShowPerRedesign && $flowSelectedRedesign !== null) {
      const p = interRedesignPeriods[$flowSelectedRedesign]
      if (p) {
        const startDate = new Date(p.start)
        const endDate = new Date(p.end)
        getTransitionMatrix(startDate, endDate, sortBy)
          .then(d => { matrixData = d })
          .catch(e => { error = e.message })
      }
    } else if (!$flowShowPerRedesign) {
      getTransitionMatrix(null, null, sortBy)
        .then(d => { matrixData = d })
        .catch(e => { error = e.message })
    }
  })

  // Auto-load design view when first displayed
  $effect(() => {
    if (viewMode === 'design' && !boardDesignData && !boardDesignLoading) {
      applyBoardDesignThresholds()
    }
  })

  // Apply button function to load board design with current thresholds
  function applyBoardDesignThresholds() {
    boardDesignLoading = true

    if ($flowShowPerRedesign && $flowSelectedRedesign !== null) {
      const p = interRedesignPeriods[$flowSelectedRedesign]
      if (p) {
        const startDate = new Date(p.start)
        const endDate = new Date(p.end)
        getBoardDesign(startDate, endDate, movementsThreshold, ccThreshold, cxThreshold, cuThreshold, movementsThreshold)
          .then(d => { boardDesignData = d })
          .catch(e => { error = e.message })
          .finally(() => { boardDesignLoading = false })
      }
    } else if (!$flowShowPerRedesign) {
      getBoardDesign(null, null, movementsThreshold, ccThreshold, cxThreshold, cuThreshold, movementsThreshold)
        .then(d => { boardDesignData = d })
        .catch(e => { error = e.message })
        .finally(() => { boardDesignLoading = false })
    }
  }

  function drawMatrix() {
    if (!svgEl || !matrixData) return

    const { row_lists, col_lists, matrix } = matrixData
    const nRows = row_lists.length
    const nCols = col_lists.length

    const cellSize  = Math.min(60, Math.max(36, Math.floor((svgEl.parentElement.clientWidth - 200) / nCols)))
    const labelPx   = Math.ceil(16 * 7 * 0.72)
    const AXIS_GAP  = 16
    const margin    = { top: labelPx + 12 + AXIS_GAP, right: 16, bottom: 16, left: 160 + AXIS_GAP }
    const W = margin.left + nCols * cellSize + margin.right
    const H = margin.top  + nRows * cellSize + margin.bottom

    svgEl.setAttribute('width',  W)
    svgEl.setAttribute('height', H)

    const maxVal = d3.max(matrix.flat()) || 1
    const color  = d3.scaleSequential(d3.interpolateBlues).domain([0, maxVal])
    const active = $selectedTransition

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    // ── Axis titles ───────────────────────────────────────────────────────────
    svg.append('text')
      .attr('transform', `translate(${AXIS_GAP}, ${margin.top + (nRows * cellSize) / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', 11).attr('font-weight', 600).attr('letter-spacing', '0.05em')
      .text('SOURCE')

    svg.append('text')
      .attr('x', margin.left + (nCols * cellSize) / 2)
      .attr('y', AXIS_GAP - 2)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted)')
      .attr('font-size', 11).attr('font-weight', 600).attr('letter-spacing', '0.05em')
      .text('TARGET')

    // ── Cells ─────────────────────────────────────────────────────────────────
    const cells = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        const v  = val || 0
        const isActive = active && active.source === row_lists[i] && active.target === col_lists[j]

        cells.append('rect')
          .attr('x', j * cellSize).attr('y', i * cellSize)
          .attr('width', cellSize - 1).attr('height', cellSize - 1)
          .attr('rx', 3)
          .attr('fill', v === 0 ? 'var(--surface-2)' : color(v))
          .attr('stroke', isActive ? 'var(--accent)' : 'none')
          .attr('stroke-width', isActive ? 2 : 0)
          .style('cursor', v > 0 ? 'pointer' : 'default')
          .on('click', () => {
            if (v === 0) return
            const next = { source: row_lists[i], target: col_lists[j] }
            const cur  = $selectedTransition
            // toggle off if same cell clicked again
            selectedTransition.set(cur && cur.source === next.source && cur.target === next.target ? null : next)
          })
          .append('title')
          .text(`${row_lists[i]} → ${col_lists[j]}: ${v}`)

        if (v > 0) {
          cells.append('text')
            .attr('x', j * cellSize + cellSize / 2).attr('y', i * cellSize + cellSize / 2)
            .attr('dy', '0.35em').attr('text-anchor', 'middle')
            .attr('fill', v > maxVal * 0.5 ? '#ffffff' : '#1a1d27')
            .attr('font-size', 11).attr('font-weight', 500)
            .attr('pointer-events', 'none')
            .text(v)
        }
      })
    })

    // ── Column labels — targets ───────────────────────────────────────────────
    const colG = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)
    col_lists.forEach((name, i) => {
      const x = i * cellSize + cellSize / 2
      const g = colG.append('g').attr('transform', `translate(${x},-8) rotate(-45)`)
      const truncated = name.length > 16 ? name.slice(0, 15) + '…' : name
      const t = g.append('text')
        .attr('text-anchor', 'start')
        .attr('fill', 'var(--text-muted)')
        .attr('font-size', 12)
        .text(truncated)
      if (name.length > 16) t.append('title').text(name)
    })

    // ── Row labels — sources ──────────────────────────────────────────────────
    const rowG = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)
    row_lists.forEach((name, i) => {
      const y = i * cellSize + cellSize / 2
      const truncated = name.length > 22 ? name.slice(0, 21) + '…' : name
      const t = rowG.append('text')
        .attr('x', -10).attr('y', y).attr('dy', '0.35em')
        .attr('text-anchor', 'end')
        .attr('fill', 'var(--text)')
        .attr('font-size', 12)
        .text(truncated)
      if (name.length > 22) t.append('title').text(name)
    })
  }
</script>

<div class="flow">
  <div class="flow-header">
    <div class="section-header">
      <h2 class="section-title">Card flow</h2>
      <p class="hint">
        {#if viewMode === 'matrix'}
          Rows = source · Columns = target · Click a cell to filter Cards tab
        {:else if viewMode === 'graph'}
          Node size = cards sent · Edge width = transition count · Drag to rearrange
        {:else}
          Board structure: lists, roles, and flow. Space+drag to pan · Space+wheel to zoom · Adjust thresholds to refine design
        {/if}
      </p>
    </div>

    <div class="controls">
      <div class="toggle">
        <button class="toggle-btn" class:active={viewMode === 'matrix'} onclick={() => viewMode = 'matrix'}>Matrix</button>
        <button class="toggle-btn" class:active={viewMode === 'graph'} onclick={() => viewMode = 'graph'}>Graph</button>
        <button class="toggle-btn" class:active={viewMode === 'design'} onclick={() => viewMode = 'design'}>Design</button>
      </div>

      {#if viewMode === 'matrix'}
        <div class="sort-control">
          <label for="sort-select">Sort by:</label>
          <select id="sort-select" bind:value={sortBy}>
            <option value="net_flow">Net flow</option>
            <option value="topological">Topological</option>
          </select>
        </div>
      {/if}

      {#if viewMode === 'design'}
        <div class="design-controls">
          <div class="sliders">
            <label class="slider-row">
              <span class="label-text">Movements:</span>
              <input type="range" min="0" max="100" bind:value={movementsThreshold} />
              <span class="value">{movementsThreshold}%</span>
            </label>
            <label class="slider-row">
              <span class="label-text">Create:</span>
              <input type="range" min="0" max="100" bind:value={ccThreshold} />
              <span class="value">{ccThreshold}%</span>
            </label>
            <label class="slider-row">
              <span class="label-text">Close:</span>
              <input type="range" min="0" max="100" bind:value={cxThreshold} />
              <span class="value">{cxThreshold}%</span>
            </label>
            <label class="slider-row">
              <span class="label-text">Use:</span>
              <input type="range" min="0" max="100" bind:value={cuThreshold} />
              <span class="value">{cuThreshold}%</span>
            </label>
          </div>
          <button class="apply-btn" onclick={applyBoardDesignThresholds} disabled={boardDesignLoading}>
            {boardDesignLoading ? 'Loading...' : 'Apply'}
          </button>
        </div>
      {/if}

      {#if interRedesignPeriods.length > 0}
        <div class="redesign-controls">
          <label>
            <input
              type="checkbox"
              checked={showPerRedesign}
              onchange={(e) => flowShowPerRedesign.set(e.target.checked)}
            />
            Between redesigns
          </label>
          {#if showPerRedesign}
            <select
              value={selectedRedesign ?? ''}
              onchange={(e) => flowSelectedRedesign.set(e.target.value === '' ? null : parseInt(e.target.value))}
            >
              <option value="">Select period…</option>
              {#each interRedesignPeriods as p, i}
                <option value={i}>
                  {new Date(p.start).toLocaleDateString()} – {new Date(p.end).toLocaleDateString()}
                </option>
              {/each}
            </select>
          {/if}
        </div>
      {/if}
    </div>
  </div>

  {#if error}
    <p class="error">{error}</p>
  {:else if viewMode === 'design'}
    {#if !boardDesignData}
      <p class="loading">Loading…</p>
    {:else}
      <BoardDesignEditor
        boardData={boardDesignData}
        readOnly={true}
        onModelChange={null}
        class="design-editor"
      />
    {/if}
  {:else if !matrixData}
    <p class="loading">Loading…</p>
  {:else if viewMode === 'matrix'}
    <div class="chart-wrap">
      <svg bind:this={svgEl} class="matrix"></svg>
    </div>
  {:else}
    <div class="chart-wrap">
      <FlowGraph {matrixData} />
    </div>
  {/if}
</div>

<style>
  .flow { display: flex; flex-direction: column; flex: 1; min-height: 0; gap: 16px; }

  .flow-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-shrink: 0;
    gap: 16px;
  }

  .section-header { display: flex; flex-direction: column; gap: 4px; }
  .section-title { font-size: 15px; font-weight: 600; }
  .hint { font-size: 12px; color: var(--text-muted); }

  .controls {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .toggle {
    display: flex;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2px;
    gap: 2px;
    flex-shrink: 0;
  }

  .toggle-btn {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    background: none;
    border-radius: calc(var(--radius) - 2px);
    transition: background 0.15s, color 0.15s;
  }

  .toggle-btn.active {
    background: var(--accent);
    color: #fff;
  }

  .redesign-controls {
    display: flex;
    gap: 8px;
    align-items: center;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2px;
    font-size: 12px;
    flex-shrink: 0;
  }

  .redesign-controls label {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    color: var(--text);
    padding: 5px 10px;
    margin: 0;
    white-space: nowrap;
  }

  .redesign-controls input[type="checkbox"] {
    cursor: pointer;
    margin: 0;
  }

  .redesign-controls select {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: calc(var(--radius) - 2px);
    padding: 5px 8px;
    font-size: 12px;
    color: var(--text);
    cursor: pointer;
    margin-right: 4px;
  }

  .sort-control {
    display: flex;
    gap: 6px;
    align-items: center;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 5px 10px;
    font-size: 12px;
    flex-shrink: 0;
  }

  .sort-control label {
    color: var(--text-muted);
    margin: 0;
    white-space: nowrap;
  }

  .sort-control select {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: calc(var(--radius) - 2px);
    padding: 4px 6px;
    font-size: 12px;
    color: var(--text);
    cursor: pointer;
  }

  .redesign-controls select:hover {
    background: var(--surface-2);
  }

  .design-controls {
    display: flex;
    gap: 8px;
    flex-direction: column;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px;
    font-size: 12px;
    flex-shrink: 0;
  }

  .design-controls {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px;
  }

  .design-controls .sliders {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex: 1;
    min-width: 0;
  }

  .design-controls label.slider-row {
    display: flex;
    align-items: center;
    gap: 3px;
    color: var(--text-muted);
    margin: 0;
    font-size: 12px;
    flex-shrink: 0;
  }

  .design-controls .label-text {
    min-width: 70px;
    font-size: 12px;
    white-space: nowrap;
  }

  .design-controls input[type="range"] {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
    width: 60px;
    height: 6px;
    padding: 0;
    margin: 0;
    cursor: pointer;
    flex-shrink: 1;
    background: transparent;
    border: none;
  }

  /* WebKit (Chrome, Safari) */
  .design-controls input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: 2px solid var(--surface);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }

  .design-controls input[type="range"]::-webkit-slider-runnable-track {
    width: 100%;
    height: 4px;
    background: var(--surface-2);
    border-radius: 2px;
    border: 1px solid var(--border);
  }

  /* Firefox */
  .design-controls input[type="range"]::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: 2px solid var(--surface);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }

  .design-controls input[type="range"]::-moz-range-track {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 2px;
    height: 4px;
  }

  .design-controls .value {
    min-width: 30px;
    max-width: 40px;
    text-align: right;
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 1;
    white-space: nowrap;
  }

  .design-controls .apply-btn {
    background: var(--accent);
    color: var(--surface);
    border: none;
    border-radius: var(--radius);
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
    white-space: nowrap;
  }

  .design-controls .apply-btn:hover:not(:disabled) {
    opacity: 0.9;
    filter: brightness(1.1);
  }

  .design-controls .apply-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px;
    overflow: auto;
    flex: 1;
    min-height: 0;
  }

  .matrix { display: block; }

  :global(.design-editor) {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 0;
    overflow: auto;
    flex: 1;
    min-height: 0;
  }

  .loading, .error { color: var(--text-muted); padding: 24px 0; }
  .error { color: var(--danger); }

  /* Small screens: adjust label width */
  @media (max-width: 600px) {
    .design-controls .label-text {
      min-width: 60px;
      font-size: 11px;
    }

    .design-controls input[type="range"] {
      width: 50px;
      min-width: 40px;
    }

    .design-controls .value {
      min-width: 28px;
      font-size: 10px;
    }
  }
</style>
