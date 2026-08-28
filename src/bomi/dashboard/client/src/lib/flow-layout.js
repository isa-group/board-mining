/**
 * Flow Graph Layout Engine
 *
 * Uses dagre library to compute hierarchical (Sugiyama) layout
 * for transition matrix graphs. Produces deterministic, aesthetically
 * pleasing layouts with minimal edge crossings.
 */

import dagre from 'dagre'

/**
 * Calculate hierarchical layout positions for a transition matrix graph.
 *
 * @param {Object} matrixData - { row_lists, col_lists, matrix }
 * @param {number} width - Canvas width in pixels
 * @param {number} height - Canvas height in pixels
 * @returns {Object} { nodes: Map<id, {x, y, _r}>, edges: Array<{source, target, waypoints}> }
 */
export function calculateFlowLayout(matrixData, width, height) {
  const { row_lists, col_lists, matrix } = matrixData

  // Collect all unique lists (nodes in the graph)
  const allLists = [...new Set([...row_lists, ...col_lists])]

  // Create dagre graph
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'LR', nodesep: 80, ranksep: 100, marginx: 60, marginy: 40 })
  g.setDefaultEdgeLabel(() => ({}))

  // Build row/col index maps
  const rowIdx = Object.fromEntries(row_lists.map((n, i) => [n, i]))
  const colIdx = Object.fromEntries(col_lists.map((n, i) => [n, i]))

  // Calculate node metrics
  const rowSums = row_lists.map((_, i) => sum(matrix[i]))
  const colSums = col_lists.map((_, j) => sum(matrix.map(r => r[j])))

  const nodes = allLists.map(name => {
    const ri = rowIdx[name] ?? -1
    const ci = colIdx[name] ?? -1
    const out = ri >= 0 ? rowSums[ri] : 0
    const inn = ci >= 0 ? colSums[ci] : 0
    return { id: name, out, inn, netFlow: out - inn }
  })

  const maxOut = Math.max(...nodes.map(d => d.out), 1)

  // Add nodes to dagre graph
  nodes.forEach(n => {
    const radius = Math.max(8, Math.min(22, 8 + 14 * (n.out / maxOut)))
    g.setNode(n.id, { label: n.id, width: radius * 2 + 50, height: radius * 2 })
  })

  // Build edge list (non-zero, no self-loops)
  const edges = []
  row_lists.forEach((src, i) => {
    col_lists.forEach((tgt, j) => {
      const count = matrix[i][j] || 0
      if (count > 0 && src !== tgt) {
        edges.push({ source: src, target: tgt, count })
        g.setEdge(src, tgt, { label: count.toString(), weight: 1 })
      }
    })
  })

  // Run dagre layout
  dagre.layout(g)

  // Extract positions from dagre
  const nodePositions = new Map()
  g.nodes().forEach(nodeId => {
    const node = g.node(nodeId)
    nodePositions.set(nodeId, {
      x: node.x,
      y: node.y,
      _r: Math.max(8, Math.min(22, 8 + 14 * ((nodes.find(n => n.id === nodeId)?.out || 0) / maxOut)))
    })
  })

  // Calculate edge data (no waypoints needed for smooth curves - handled in rendering)
  const edgeData = new Map()
  edges.forEach(edge => {
    const key = `${edge.source}-${edge.target}`
    const sourceNode = nodePositions.get(edge.source)
    const targetNode = nodePositions.get(edge.target)

    if (sourceNode && targetNode) {
      edgeData.set(key, {
        source: sourceNode,
        target: targetNode,
        count: edge.count
      })
    }
  })

  return {
    nodes: nodePositions,
    edges,
    edgeData,
    dimensions: { width, height }
  }
}

/**
 * Helper: sum array
 */
function sum(arr) {
  return arr.reduce((a, b) => a + b, 0)
}
