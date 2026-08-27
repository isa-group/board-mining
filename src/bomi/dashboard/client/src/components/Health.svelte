<script>
  import { onMount } from 'svelte'
  import { getHealthEvolution } from '../api.js'
  import { healthEvolutionCache, selectedPeriod } from '../stores.js'
  import * as d3 from 'd3'

  let data = $state(null)
  let error = $state(null)
  let svgEl = $state(null)
  let loading = $state(false)

  const DIM_COLORS = {
    dim_flow_discipline:         '#6c8ef5',
    dim_collaboration_discipline:'#52c77f',
    dim_completion_discipline:   '#f0a500',
    dim_structural_stability:    '#c77bff',
    dim_board_vitality:          '#f56c6c',
  }

  $effect(() => {
    if (data && svgEl) drawChart()
  })

  onMount(async () => {
    const cached = $healthEvolutionCache
    if (cached) { data = cached; return }

    loading = true
    try {
      const raw = await getHealthEvolution(30, 7, Object.keys(DIM_COLORS))
      healthEvolutionCache.set(raw)
      data = raw
    } catch (e) {
      error = e.message
    } finally {
      loading = false
    }
  })

  function drawChart() {
    if (!svgEl || !data?.length) return

    const margin = { top: 20, right: 130, bottom: 40, left: 44 }
    const W = svgEl.clientWidth || 800
    const H = 300

    svgEl.setAttribute('height', H)

    const parseDate = d3.isoParse
    const dims = Object.keys(DIM_COLORS)

    const x = d3.scaleTime()
      .domain(d3.extent(data, d => parseDate(d.timestamp)))
      .range([margin.left, W - margin.right])

    const y = d3.scaleLinear().domain([0, 1]).range([H - margin.bottom, margin.top])

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    const line = d3.line()
      .x(d => x(parseDate(d.timestamp)))
      .defined(d => d.value != null)
      .y(d => y(d.value))
      .curve(d3.curveCatmullRom)

    dims.forEach(dim => {
      const series = data.map(d => ({ timestamp: d.timestamp, value: d[dim] }))
      svg.append('path')
        .datum(series)
        .attr('fill', 'none')
        .attr('stroke', DIM_COLORS[dim])
        .attr('stroke-width', 2)
        .attr('d', line)
    })

    // Selected period overlay
    const period = $selectedPeriod
    if (period) {
      svg.append('rect')
        .attr('x', x(period[0]))
        .attr('y', margin.top)
        .attr('width', x(period[1]) - x(period[0]))
        .attr('height', H - margin.top - margin.bottom)
        .attr('fill', 'rgba(108, 142, 245, 0.12)')
        .attr('pointer-events', 'none')
    }

    // Axes
    svg.append('g')
      .attr('transform', `translate(0,${H - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.timeFormat('%b %Y')))
      .call(g => g.select('.domain').attr('stroke', 'var(--border)'))
      .call(g => g.selectAll('text').attr('fill', 'var(--text-muted)').attr('font-size', 11))
      .call(g => g.selectAll('line').attr('stroke', 'var(--border)'))

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('.0%')))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('text').attr('fill', 'var(--text-muted)').attr('font-size', 11))
      .call(g => g.selectAll('line').attr('stroke', 'var(--border)'))

    // Legend
    const legend = svg.append('g').attr('transform', `translate(${W - margin.right + 12}, ${margin.top})`)
    dims.forEach((dim, i) => {
      const g = legend.append('g').attr('transform', `translate(0, ${i * 22})`)
      g.append('line').attr('x1', 0).attr('x2', 14).attr('y1', 6).attr('y2', 6)
        .attr('stroke', DIM_COLORS[dim]).attr('stroke-width', 2)
      g.append('text').attr('x', 18).attr('y', 10)
        .attr('fill', 'var(--text-muted)').attr('font-size', 10)
        .text(dim.replace('dim_', '').replace(/_/g, ' '))
    })
  }
</script>

<div class="health">
  <div class="section-header">
    <h2 class="section-title">Health evolution over time</h2>
    <p class="hint">Dimensions computed with a 30-day rolling window · step 7 days</p>
  </div>

  {#if error}
    <p class="error">{error}</p>
  {:else if loading}
    <p class="loading">Computing health evolution…</p>
  {:else if !data}
    <p class="loading">Loading…</p>
  {:else}
    <div class="chart-wrap">
      <svg bind:this={svgEl} class="evolution" width="100%"></svg>
    </div>

    <div class="dimensions-info">
      <div class="dimension-card">
        <div class="dim-color" style="background: {DIM_COLORS.dim_flow_discipline}"></div>
        <div class="dim-text">
          <h3>Flow discipline</h3>
          <p>Card movements respect the board's intended flow. Combines adherence to prescribed transitions, low bouncing (re-entries to previous lists), and minimal silent moves without activity.</p>
        </div>
      </div>

      <div class="dimension-card">
        <div class="dim-color" style="background: {DIM_COLORS.dim_collaboration_discipline}"></div>
        <div class="dim-text">
          <h3>Collaboration discipline</h3>
          <p>Cards are properly managed and assigned. Measures the absence of orphaned cards (lacking creation or movement records) and the presence of team assignment.</p>
        </div>
      </div>

      <div class="dimension-card">
        <div class="dim-color" style="background: {DIM_COLORS.dim_completion_discipline}"></div>
        <div class="dim-text">
          <h3>Completion discipline</h3>
          <p>Work is finished and delivered consistently. Combines completion rate (cards closed relative to created) and low abandonment (work explicitly discarded).</p>
        </div>
      </div>

      <div class="dimension-card">
        <div class="dim-color" style="background: {DIM_COLORS.dim_structural_stability}"></div>
        <div class="dim-text">
          <h3>Structural stability</h3>
          <p>Board structure remains consistent and functional. Measured by absence of dead lists (no recent activity) and minimal reorganizations.</p>
        </div>
      </div>

      <div class="dimension-card">
        <div class="dim-color" style="background: {DIM_COLORS.dim_board_vitality}"></div>
        <div class="dim-text">
          <h3>Board vitality</h3>
          <p>Work flows actively through the board. Combines low rates of inactive cards (stalled for extended periods) and stagnant lists (receiving no recent work).</p>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .health { display: flex; flex-direction: column; gap: 20px; flex: 1; overflow-y: auto; }
  .section-header { display: flex; flex-direction: column; gap: 4px; }
  .section-title { font-size: 15px; font-weight: 600; }
  .hint { font-size: 12px; color: var(--text-muted); }
  .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px; overflow-x: auto; }
  .evolution { display: block; }
  .loading, .error { color: var(--text-muted); padding: 24px 0; }
  .error { color: var(--danger); }

  /* Dimensions info */
  .dimensions-info {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
  }

  .dimension-card {
    display: flex;
    gap: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 12px;
  }

  .dim-color {
    flex: 0 0 4px;
    border-radius: 2px;
  }

  .dim-text {
    flex: 1;
    min-width: 0;
  }

  .dim-text h3 {
    font-size: 12px;
    font-weight: 600;
    margin: 0 0 4px 0;
    color: var(--text);
    text-transform: capitalize;
  }

  .dim-text p {
    font-size: 11px;
    line-height: 1.4;
    color: var(--text-muted);
    margin: 0;
  }
</style>
