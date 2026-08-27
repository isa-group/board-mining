<script>
  import { getListEvolution, getRedesigns, getEventsByList } from '../api.js'
  import { selectedPeriod, selectedList, redesignThresholdDays, redesignThresholdLEvents, redesignPeriods } from '../stores.js'
  import * as d3 from 'd3'

  let lists = $state([])
  let redesigns = $state([])
  let events = $state([])
  let error = $state(null)
  let svgEl = $state(null)
  let showEventScatter = $state(false)
  let selectedCardId = $state(null)

  // Bind to stores so values persist across tabs
  let thresholdDays = $derived($redesignThresholdDays)
  let thresholdLEvents = $derived($redesignThresholdLEvents)

  $effect(() => {
    if (lists.length && svgEl) {
      // Remove old context SVG if it exists
      const oldContextSvg = document.querySelector('.context-pane-svg')
      if (oldContextSvg) oldContextSvg.remove()
      drawGantt()
    }
  })

  async function loadData() {
    try {
      const [l, r, ev] = await Promise.all([
        getListEvolution(),
        getRedesigns($redesignThresholdDays, null, $redesignThresholdLEvents),
        getEventsByList(),
      ])
      lists = l
      redesigns = r
      events = ev
      redesignPeriods.set(r)  // Cache in store for Flow tab to use
    } catch (e) {
      error = e.message
    }
  }

  async function reloadRedesigns() {
    try {
      const r = await getRedesigns($redesignThresholdDays, null, $redesignThresholdLEvents)
      redesigns = r
      redesignPeriods.set(r)  // Cache in store for Flow tab to use
    } catch (e) {
      error = e.message
    }
  }

  loadData()

  function drawGantt() {
    if (!svgEl || !lists.length) return

    // Sort lists by begin_date, then by last_date
    const sortedLists = [...lists].sort((a, b) => {
      const aBegin = new Date(a.begin_date)
      const bBegin = new Date(b.begin_date)
      if (aBegin !== bBegin) return aBegin - bBegin
      const aLast = new Date(a.last_date)
      const bLast = new Date(b.last_date)
      return aLast - bLast
    })

    // Calculate dynamic left margin based on longest label
    const maxNameLength = Math.max(...sortedLists.map(d => (d.last_name ?? d.list_id).length))
    const dynamicLeftMargin = Math.max(160, Math.ceil(maxNameLength * 7) + 16)

    const margin = { top: 20, right: 20, bottom: 40, left: dynamicLeftMargin }
    // Get width from parent container and account for padding
    const containerWidth = svgEl.parentElement?.clientWidth || 800
    const W = Math.max(600, containerWidth - 32) // Subtract padding from chart-wrap
    const rowH = 28
    const H = sortedLists.length * rowH + margin.top + margin.bottom

    svgEl.setAttribute('width', W)
    svgEl.setAttribute('height', H)

    // Helper to parse dates — handles both strings and Date objects
    const parseDate = (dateVal) => {
      if (!dateVal) return null
      if (dateVal instanceof Date) return dateVal
      return d3.isoParse(dateVal)
    }
    const allDates = sortedLists.flatMap(d => [parseDate(d.begin_date), parseDate(d.last_date)].filter(Boolean))
    const xExtent = d3.extent(allDates)

    const x = d3.scaleTime().domain(xExtent).range([margin.left, W - margin.right])
    const y = d3.scaleBand()
      .domain(sortedLists.map(d => d.list_id))
      .range([margin.top, H - margin.bottom])
      .padding(0.25)

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    // Create a group for zoomable content
    const g = svg.append('g').attr('class', 'zoomable-group')

    // Store current x-scale for redrawing
    let currentXScale = x

    // Redesign period overlays (columns: min, max)
    g.append('g').attr('class', 'redesigns').selectAll('rect')
      .data(redesigns)
      .join('rect')
        .attr('x', d => x(parseDate(d.min)))
        .attr('y', margin.top)
        .attr('width', d => Math.max(0, x(parseDate(d.max)) - x(parseDate(d.min))))
        .attr('height', H - margin.top - margin.bottom)
        .attr('fill', 'rgba(108, 142, 245, 0.07)')
        .attr('stroke', 'rgba(108, 142, 245, 0.25)')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
          selectedPeriod.set([parseDate(d.min), parseDate(d.max)])
        })

    // List bars (columns: begin_date, last_date)
    g.append('g').attr('class', 'lists').selectAll('rect')
      .data(sortedLists)
      .join('rect')
        .attr('x', d => x(parseDate(d.begin_date)))
        .attr('y', d => y(d.list_id))
        .attr('width', d => Math.max(2, x(parseDate(d.last_date)) - x(parseDate(d.begin_date))))
        .attr('height', y.bandwidth())
        .attr('rx', 4)
        .attr('fill', 'var(--accent-dim)')
        .attr('stroke', 'var(--accent)')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
          selectedList.set(d.list_id === $selectedList ? null : d.list_id)
        })

    // Y-axis labels (last_name = most recent list name)
    g.append('g').attr('class', 'labels')
      .selectAll('text')
      .data(sortedLists)
      .join('text')
        .attr('x', margin.left - 8)
        .attr('y', d => y(d.list_id) + y.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'end')
        .attr('fill', d => d.list_id === $selectedList ? 'var(--accent)' : 'var(--text)')
        .attr('font-size', 12)
        .text(d => d.last_name ?? d.list_id)

    // Event scatter plot (optional overlay)
    if (showEventScatter && events.length) {
      const eventColors = {
        card_create: '#52c77f',
        card_act: '#6c8ef5',
        card_move: '#f0a500',
        card_close: '#f56c6c',
      }

      const jitterAmount = y.bandwidth() * 0.15 // 30% spread, so ±15%
      const jitterSeed = (d, i) => {
        // Deterministic jitter based on event index and card_id
        const seed = (d.card_id || 'default').charCodeAt(0) || 0
        return (Math.sin(i * 12.9898 + seed) * 43758.5453) % 1
      }

      g.append('g').attr('class', 'scatter')
        .selectAll('circle')
        .data(events)
        .join('circle')
          .attr('cx', d => x(parseDate(d.timestamp)))
          .attr('cy', d => {
            const bandCenter = y(d.list_id) + y.bandwidth() / 2
            const jitter = (jitterSeed(d, Math.random()) - 0.5) * jitterAmount * 2
            return bandCenter + jitter
          })
          .attr('r', 2.5)
          .attr('fill', d => eventColors[d.card_event_type] || '#999')
          .attr('opacity', 0.7)
          .attr('stroke', 'none')
          .attr('stroke-width', 1.5)
          .style('cursor', 'pointer')
          .style('display', d => selectedCardId && selectedCardId !== d.card_id ? 'none' : 'block')
          .on('click', (event, d) => {
            event.stopPropagation()
            selectedCardId = selectedCardId === d.card_id ? null : d.card_id
            // Update all points visibility and styling
            d3.selectAll('.scatter circle')
              .style('display', (point) => selectedCardId && selectedCardId !== point.card_id ? 'none' : 'block')
              .attr('stroke', (point) => selectedCardId === point.card_id ? 'var(--accent)' : 'none')
              .attr('opacity', (point) => selectedCardId === point.card_id ? 1.0 : 0.7)
          })
          .on('mouseenter', function(event, d) {
            d3.select(this).attr('stroke', 'var(--accent)').attr('opacity', 1.0)
            // Show tooltip
            const tooltip = d3.select('body').append('div')
              .attr('class', 'event-tooltip')
              .style('position', 'absolute')
              .style('background', 'var(--surface-2)')
              .style('border', '1px solid var(--border)')
              .style('border-radius', 'var(--radius)')
              .style('padding', '8px 12px')
              .style('font-size', '12px')
              .style('color', 'var(--text)')
              .style('pointer-events', 'none')
              .style('z-index', '1000')
              .style('left', (event.pageX + 10) + 'px')
              .style('top', (event.pageY - 10) + 'px')
              .html(`${d.card_name ? d.card_name + ' ' : ''}#${d.card_id}<br/>
                     List: ${d.list_name || d.list_id || 'unknown'}<br/>
                     ${d.card_event_type}<br/>
                     ${new Date(d.timestamp).toLocaleString()}`)
            // Store reference for cleanup
            d3.select(this).attr('data-tooltip', 'active')
          })
          .on('mouseleave', function(event, d) {
            const isSelected = selectedCardId === d.card_id
            d3.select(this)
              .attr('stroke', isSelected ? 'var(--accent)' : 'none')
              .attr('opacity', isSelected ? 1.0 : 0.7)
            // Remove tooltip
            d3.selectAll('.event-tooltip').remove()
          })

      // Click on chart background to deselect
      svg.on('click', () => {
        selectedCardId = null
        d3.selectAll('.scatter circle')
          .style('display', 'block')
          .attr('opacity', 0.7)
          .attr('stroke', 'none')
      })
    }

    // X-axis (outside zoomable group so it stays fixed)
    const xAxisG = svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${H - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.timeFormat('%b %Y')))
      .call(g => g.select('.domain').attr('stroke', 'var(--border)'))
      .call(g => g.selectAll('text').attr('fill', 'var(--text-muted)').attr('font-size', 11))
      .call(g => g.selectAll('line').attr('stroke', 'var(--border)'))

    // Helper function to redraw main chart with a new x-scale domain
    const redrawMainChart = (x0, x1) => {
      const newX = d3.scaleTime().domain([x0, x1]).range([margin.left, W - margin.right])

      // Update redesign overlays
      g.select('.redesigns').selectAll('rect')
        .attr('x', d => {
          const minDate = parseDate(d.min)
          return minDate ? newX(minDate) : margin.left
        })
        .attr('width', d => {
          const minDate = parseDate(d.min)
          const maxDate = parseDate(d.max)
          if (!minDate || !maxDate) return 0
          const w = newX(maxDate) - newX(minDate)
          return isNaN(w) ? 0 : Math.max(0, w)
        })

      // Update list bars
      g.select('.lists').selectAll('rect')
        .attr('x', d => {
          const beginDate = parseDate(d.begin_date)
          return beginDate ? newX(beginDate) : margin.left
        })
        .attr('width', d => {
          const beginDate = parseDate(d.begin_date)
          const lastDate = parseDate(d.last_date)
          if (!beginDate || !lastDate) return 0
          const w = newX(lastDate) - newX(beginDate)
          return isNaN(w) ? 0 : Math.max(2, w)
        })

      // Update scatter points
      g.select('.scatter').selectAll('circle')
        .attr('cx', d => {
          const ts = parseDate(d.timestamp)
          return ts ? newX(ts) : margin.left
        })
        // Preserve selection state during zoom
        .style('display', d => selectedCardId && selectedCardId !== d.card_id ? 'none' : 'block')
        .attr('opacity', d => selectedCardId === d.card_id ? 1.0 : 0.7)
        .attr('stroke', d => selectedCardId === d.card_id ? 'var(--accent)' : 'none')

      // Update X-axis with new scale
      xAxisG.call(d3.axisBottom(newX).ticks(6).tickFormat(d3.timeFormat('%b %Y')))
        .call(g => g.select('.domain').attr('stroke', 'var(--border)'))
        .call(g => g.selectAll('text').attr('fill', 'var(--text-muted)').attr('font-size', 11))
        .call(g => g.selectAll('line').attr('stroke', 'var(--border)'))
    }

    // Context pane setup
    const CONTEXT_HEIGHT = 70
    const contextMargin = { top: 15, right: 20, bottom: 25, left: dynamicLeftMargin }

    // Remove any existing context SVG
    const existingContext = document.querySelector('.context-pane-svg')
    if (existingContext) existingContext.remove()

    // Create context SVG with explicit dimensions (sticky so it doesn't scroll away)
    const contextSvgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    contextSvgEl.setAttribute('class', 'context-pane-svg')
    contextSvgEl.setAttribute('width', W)
    contextSvgEl.setAttribute('height', CONTEXT_HEIGHT)
    contextSvgEl.setAttribute('style', 'display: block; margin-top: 12px; background: var(--surface-2); border-radius: var(--radius-lg); position: sticky; bottom: 0; z-index: 10;')
    svgEl.parentElement.appendChild(contextSvgEl)

    const contextSvg = d3.select(contextSvgEl)
    const contextX = d3.scaleTime().domain(xExtent).range([contextMargin.left, W - contextMargin.right])
    const contextG = contextSvg.append('g')

    // Draw vertical lines for redesign periods
    contextG.selectAll('line.redesign-line')
      .data(redesigns)
      .join('line')
        .attr('class', 'redesign-line')
        .attr('x1', d => contextX(parseDate(d.min)))
        .attr('x2', d => contextX(parseDate(d.min)))
        .attr('y1', contextMargin.top)
        .attr('y2', CONTEXT_HEIGHT - contextMargin.bottom)
        .attr('stroke', 'rgba(108, 142, 245, 0.5)')
        .attr('stroke-width', 2)

    // Context X-axis
    contextG.append('g')
      .attr('transform', `translate(0,${CONTEXT_HEIGHT - contextMargin.bottom})`)
      .call(d3.axisBottom(contextX).ticks(4).tickFormat(d3.timeFormat('%b %Y')))
      .call(g => g.select('.domain').attr('stroke', 'var(--border)'))
      .call(g => g.selectAll('text').attr('fill', 'var(--text-muted)').attr('font-size', 10))
      .call(g => g.selectAll('line').attr('stroke', 'var(--border)'))

    // Brush setup
    const brush = d3.brush()
      .extent([[contextMargin.left, contextMargin.top], [W - contextMargin.right, CONTEXT_HEIGHT - contextMargin.bottom]])
      .on('brush', (event) => {
        if (event.selection) {
          const [[x0Px, y0], [x1Px, y1]] = event.selection
          const x0 = contextX.invert(x0Px)
          const x1 = contextX.invert(x1Px)
          if (x0 && x1 && !isNaN(x0) && !isNaN(x1)) {
            redrawMainChart(x0, x1)
          }
        }
      })

    const brushG = contextG.append('g')
      .attr('class', 'brush')
      .call(brush)

    // Set initial brush selection to full range (defer to avoid race condition)
    setTimeout(() => {
      brushG.call(brush.move, [[contextMargin.left, contextMargin.top], [W - contextMargin.right, CONTEXT_HEIGHT - contextMargin.bottom]])
    }, 0)
  }
</script>

<div class="structure">
  <div class="structure-header">
    <div class="section-header">
      <h2 class="section-title">List lifecycle &amp; redesign periods</h2>
      <p class="hint">
        Click a bar to filter by list · Use the selector below to zoom and pan the timeline
      </p>
    </div>

    <div class="params-section">
      <h3 class="params-title">Visualization & Detection</h3>
      <div class="params-controls">
        <div class="control-group">
          <label>
            <input
              type="checkbox"
              checked={showEventScatter}
              onchange={(e) => showEventScatter = e.target.checked}
            />
            Show event scatter
          </label>
        </div>
        <div class="control-group">
          <label for="threshold-days">Time window (days):</label>
          <input
            id="threshold-days"
            type="number"
            value={thresholdDays}
            onchange={(e) => redesignThresholdDays.set(parseInt(e.target.value) || 1)}
            onblur={reloadRedesigns}
            min="1"
            max="30"
            title="Time window in days for detecting redesigns"
          />
        </div>
        <div class="control-group">
          <label for="threshold-events">Min list events:</label>
          <input
            id="threshold-events"
            type="number"
            value={thresholdLEvents}
            onchange={(e) => redesignThresholdLEvents.set(parseInt(e.target.value) || 0)}
            onblur={reloadRedesigns}
            min="0"
            max="100"
            title="Minimum list events required to report redesign"
          />
        </div>
      </div>
    </div>
  </div>

  {#if error}
    <p class="error">{error}</p>
  {:else if !lists.length}
    <p class="loading">Loading…</p>
  {:else}
    <div class="chart-wrap">
      <svg bind:this={svgEl} class="gantt" width="100%"></svg>
    </div>
    {#if redesigns.length}
      <p class="hint">Shaded regions: {redesigns.length} redesign period{redesigns.length !== 1 ? 's' : ''} detected</p>
    {/if}
  {/if}
</div>

<style>
  .structure { display: flex; flex-direction: column; gap: 20px; flex: 1; overflow-y: auto; }

  .structure-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-shrink: 0;
    gap: 20px;
  }

  .section-header { display: flex; flex-direction: column; gap: 4px; flex: 1; }
  .section-title { font-size: 15px; font-weight: 600; }
  .hint { font-size: 12px; color: var(--text-muted); }

  .params-section {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .params-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
  }

  .params-controls {
    display: flex;
    flex-direction: row;
    gap: 12px;
    align-items: center;
    padding: 0;
    flex-wrap: wrap;
  }

  .control-group {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .control-group label {
    font-size: 11px;
    color: var(--text-muted);
    white-space: nowrap;
  }

  .control-group input {
    width: 50px;
    padding: 3px 5px;
    font-size: 11px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: calc(var(--radius) - 1px);
    color: var(--text);
  }

  .control-group input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(108, 142, 245, 0.2);
  }

  .control-group input[type="checkbox"] {
    width: auto;
    cursor: pointer;
    margin-right: 4px;
  }

  .control-group label {
    display: flex;
    align-items: center;
    cursor: pointer;
  }

  .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px; overflow-x: auto; }
  .gantt { display: block; }
  .loading, .error { color: var(--text-muted); padding: 24px 0; }
  .error { color: var(--danger); }

  /* Brush selector styles */
  :global(.brush .selection) {
    fill: rgba(108, 142, 245, 0.15);
    stroke: var(--accent);
    stroke-width: 1;
  }

  :global(.brush .handle) {
    fill: var(--accent);
    cursor: ew-resize;
    stroke: none;
  }

  :global(.brush .handle:hover) {
    fill: var(--accent);
    opacity: 0.8;
  }
</style>
