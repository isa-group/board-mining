<script>
  import { selectedTransition } from '../stores.js'
  import { calculateFlowLayout } from '../lib/flow-layout.js'
  import * as d3 from 'd3'

  let { matrixData } = $props()

  let svgEl = $state(null)

  $effect(() => {
    if (svgEl && matrixData) draw()
  })

  function draw() {
    // ── Layout ────────────────────────────────────────────────────────────────
    const W = Math.max(400, svgEl.parentElement.clientWidth  - 32)
    const H = Math.max(300, svgEl.parentElement.clientHeight - 32)
    svgEl.setAttribute('width',  W)
    svgEl.setAttribute('height', H)

    // Calculate hierarchical layout using dagre
    const layout = calculateFlowLayout(matrixData, W, H)
    const { nodes: nodePositions, edges, edgeData } = layout

    // Rebuild node data with positions and metrics
    const allLists = [...nodePositions.keys()]
    const { row_lists, col_lists, matrix } = matrixData
    const rowIdx = Object.fromEntries(row_lists.map((n, i) => [n, i]))
    const colIdx = Object.fromEntries(col_lists.map((n, i) => [n, i]))

    const rowSums = row_lists.map((_, i) => d3.sum(matrix[i]))
    const colSums = col_lists.map((_, j) => d3.sum(matrix.map(r => r[j])))

    /**
     * Generate smooth Bezier curve path between two nodes
     * Routes edges based on actual direction vector
     * Distributes multiple edges around node perimeter by rotating the angle
     * Edges start and end exactly on circle borders
     */
    const curvePath = (source, target, edgeIndex = 0, edgeCount = 1) => {
      // Calculate direction vector from source to target
      const dx = target.x - source.x
      const dy = target.y - source.y
      const len = Math.sqrt(dx * dx + dy * dy) || 1
      const angle = Math.atan2(dy, dx)

      // For multiple edges, rotate the angle slightly to distribute around perimeter
      const angleSpacing = 0.15  // radians, roughly 8-10 degrees per edge
      const angleOffset = (edgeIndex - (edgeCount - 1) / 2) * angleSpacing
      const rotatedAngle = angle + angleOffset

      // Calculate rotated direction vector
      const dirX = Math.cos(rotatedAngle)
      const dirY = Math.sin(rotatedAngle)

      // Exit source node exactly on circle border at rotated angle
      let x1 = source.x + dirX * source._r
      let y1 = source.y + dirY * source._r

      // Enter target node with clearance for arrowhead
      // End slightly before circle so arrowhead extends to/past border
      const arrowheadSize = 5  // pixels clearance for arrowhead
      let x2 = target.x - dirX * (target._r + arrowheadSize)
      let y2 = target.y - dirY * (target._r + arrowheadSize)

      // Control point at midpoint, offset perpendicular to edge direction
      const mx = (x1 + x2) / 2
      const my = (y1 + y2) / 2
      const edgeDx = x2 - x1
      const edgeDy = y2 - y1
      const edgeLen = Math.sqrt(edgeDx * edgeDx + edgeDy * edgeDy) || 1

      // Perpendicular offset for smooth curve
      const curveOffset = Math.min(80, edgeLen * 0.3)
      const cx = mx - edgeDy / edgeLen * curveOffset
      const cy = my + edgeDx / edgeLen * curveOffset

      return `M${x1},${y1}Q${cx},${cy} ${x2},${y2}`
    }

    const nodes = allLists.map(name => {
      const ri = rowIdx[name] ?? -1
      const ci = colIdx[name] ?? -1
      const out = ri >= 0 ? rowSums[ri] : 0
      const inn = ci >= 0 ? colSums[ci] : 0
      const pos = nodePositions.get(name)
      return {
        id: name,
        x: pos.x,
        y: pos.y,
        out,
        inn,
        netFlow: out - inn,
        _r: pos._r
      }
    })

    // Reindex edges with node objects for consistency
    const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]))
    const simEdges = edges.map(e => ({
      source: nodeById[e.source],
      target: nodeById[e.target],
      sourceId: e.source,
      targetId: e.target,
      count: e.count
    }))

    const maxCount = d3.max(simEdges, d => d.count) || 1
    const edgeW    = d => 1.5 + 4   * Math.sqrt(d.count / maxCount)
    const edgeOpac = d => 0.25 + 0.6 * (d.count / maxCount)

    // Group edges by target to assign indices for distribution
    const edgesByTarget = {}
    const edgesBySource = {}
    simEdges.forEach(edge => {
      const targetKey = edge.target.id
      const sourceKey = edge.source.id

      if (!edgesByTarget[targetKey]) edgesByTarget[targetKey] = []
      if (!edgesBySource[sourceKey]) edgesBySource[sourceKey] = []

      edgesByTarget[targetKey].push(edge)
      edgesBySource[sourceKey].push(edge)
    })

    // Assign indices to edges within their groups
    simEdges.forEach(edge => {
      const targetKey = edge.target.id
      const sourceKey = edge.source.id

      // Use target grouping for index (multiple edges to same target)
      edge._targetIndex = edgesByTarget[targetKey].indexOf(edge)
      edge._targetCount = edgesByTarget[targetKey].length
    })

    // ── SVG ───────────────────────────────────────────────────────────────────
    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()

    // Add a background rect for zoom/pan interaction
    svg.append('rect')
      .attr('width', W)
      .attr('height', H)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')

    // Create main group for content (will be transformed by zoom)
    const mainGroup = svg.append('g')
      .attr('class', 'zoom-group')

    // Add zoom behavior
    const zoomBehavior = d3.zoom()
      .on('zoom', (event) => {
        mainGroup.attr('transform', event.transform)
      })

    svg.call(zoomBehavior)

    // Add marker definitions to main group
    mainGroup.append('defs').append('marker')
      .attr('id', 'dfg-arrow')
      .attr('viewBox', '0 -4 10 8')
      .attr('refX', 5).attr('refY', 0)
      .attr('markerWidth', 3).attr('markerHeight', 3)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,-4 L10,0 L0,4 Z')
      .attr('fill', 'var(--accent)')

    // ── Edges ──────────────────────────────────────────────────────────────────
    const edgeG     = mainGroup.append('g')
    const edgeGroup = edgeG.selectAll('g')
      .data(simEdges)
      .join('g')
      .style('cursor', 'pointer')
      .on('click', (_, d) => {
        const cur = $selectedTransition
        selectedTransition.set(
          cur && cur.source === d.sourceId && cur.target === d.targetId
            ? null
            : { source: d.sourceId, target: d.targetId }
        )
      })

    const paths = edgeGroup.append('path')
      .attr('fill', 'none')
      .attr('stroke', 'var(--accent)')
      .attr('stroke-width', d => edgeW(d))
      .attr('stroke-opacity', d => edgeOpac(d))
      .attr('marker-end', 'url(#dfg-arrow)')
      .attr('d', d => curvePath(d.source, d.target, d._targetIndex, d._targetCount))

    const edgeLabels = edgeGroup.append('text')
      .attr('class', 'edge-label')
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text)')
      .attr('font-size', 11).attr('font-weight', 600)
      .text(d => d.count)
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2 - 10)

    // ── Nodes ──────────────────────────────────────────────────────────────────
    const nodeG      = mainGroup.append('g')
    const nodeGroups = nodeG.selectAll('g')
      .data(nodes)
      .join('g')
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .style('cursor', 'grab')

    nodeGroups.append('circle')
      .attr('r', d => d._r)
      .attr('fill',         d => d.netFlow > 0 ? 'var(--accent)' : d.netFlow < 0 ? 'var(--text-muted)' : 'var(--accent-dim)')
      .attr('stroke',       'var(--bg)')
      .attr('stroke-width', 2)

    // Helper: wrap text into lines based on max width (chars)
    const wrapText = (text, maxChars = 18) => {
      const words = text.split(' ')
      const lines = []
      let currentLine = ''

      words.forEach(word => {
        const testLine = currentLine + (currentLine ? ' ' : '') + word
        if (testLine.length > maxChars && currentLine) {
          lines.push(currentLine)
          currentLine = word
        } else {
          currentLine = testLine
        }
      })
      if (currentLine) lines.push(currentLine)
      return lines
    }

    // Label pill (sized after text is in DOM)
    nodeGroups.append('rect').attr('class', 'label-pill').attr('rx', 10).attr('height', 20).attr('y', -10)

    // Wrapped text label with tspan for each line
    const textElements = nodeGroups.append('text')
      .attr('class', 'node-label')
      .attr('font-size', 12).attr('font-weight', 500)
      .attr('fill', 'var(--text)')
      .attr('text-anchor', 'middle')

    // Position pill and label once text bounding boxes are available
    nodeGroups.each(function(d) {
      const g   = d3.select(this)
      const lines = wrapText(d.id)
      const pad = 8
      const px  = d._r + 6

      // Measure text dimensions
      let maxWidth = 0
      lines.forEach(line => {
        const tempText = g.append('text')
          .attr('font-size', 12).attr('font-weight', 500)
          .attr('visibility', 'hidden')
          .text(line)

        const bbox = tempText.node()?.getBBox()
        maxWidth = Math.max(maxWidth, bbox?.width || 0)
        tempText.remove()
      })

      const tw  = maxWidth || 60
      const th  = lines.length * 14  // lineHeight * number of lines
      const pillCenter = px + pad + tw / 2

      // Now create the actual text with tspans positioned at pill center
      const textElement = g.select('.node-label')
      textElement.text('')

      const lineHeight = 14
      const startY = -(lines.length - 1) * lineHeight / 2

      lines.forEach((line, i) => {
        textElement.append('tspan')
          .attr('x', pillCenter)  // Position in pill, not at node center
          .attr('y', startY + i * lineHeight)
          .attr('dy', i === 0 ? '0.35em' : 0)
          .text(line)
      })

      // Position pill around text
      g.select('.label-pill')
        .attr('x',     px)
        .attr('y',     -th / 2 - 2)
        .attr('width', tw + pad * 2)
        .attr('height', th + 4)
        .attr('fill',  d.netFlow > 0 ? 'var(--accent)' : d.netFlow < 0 ? 'var(--text-muted)' : 'var(--accent-dim)')
        .attr('fill-opacity', 0.15)
    })
  }
</script>

<svg bind:this={svgEl} class="dfg"></svg>

<style>
  .dfg { display: block; width: 100%; height: 100%; }
  :global(.edge-label)                   { opacity: 0; transition: opacity 0.15s; pointer-events: none; }
  :global(.edge-group:hover .edge-label) { opacity: 1; }
</style>
