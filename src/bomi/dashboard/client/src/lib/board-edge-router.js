/**
 * Board Edge Router - Orthogonal (Manhattan) routing for semantic precedence edges
 *
 * Generates waypoints for edges that don't overlap, using orthogonal paths.
 * - Adjacent boxes: straight horizontal line
 * - Non-adjacent left-to-right: up → right → down
 * - Non-adjacent right-to-left: down → left → up
 */

export function calculateEdgeRouting(sourceShape, targetShape, edgeIndex, totalEdges, allShapes) {
  if (!sourceShape || !targetShape) return null

  // Calculate edge entry/exit points (not centers)
  const sourceY = sourceShape.y + sourceShape.height / 2
  const targetY = targetShape.y + targetShape.height / 2
  const sourceX = sourceShape.x + sourceShape.width
  const targetX = targetShape.x

  // Detect if boxes are adjacent (no other boxes between them)
  const isAdjacent = areAdjacent(sourceShape, targetShape, allShapes)

  // For adjacent boxes, use a simple straight line
  if (isAdjacent) {
    return [
      { x: sourceX, y: sourceY },
      { x: targetX, y: targetY },
    ]
  }

  // Calculate vertical offset to avoid overlaps
  // List boxes are 80px tall, so we need significant spacing between edges
  const centerIndex = Math.floor(totalEdges / 2)
  const edgeSpacing = 60 // pixels per edge level - ensures clear separation between edges
  const baseOffset = 80 // minimum distance above/below the listbox
  const offset = (edgeIndex - centerIndex) * edgeSpacing

  // For non-adjacent boxes, use orthogonal routing
  const isLeftToRight = sourceShape.x < targetShape.x

  if (isLeftToRight) {
    // Left-to-right: up → right → down
    const controlY = sourceY - baseOffset - offset

    return [
      { x: sourceX, y: sourceY },
      { x: sourceX + 10, y: sourceY },
      { x: sourceX + 10, y: controlY },
      { x: targetX - 10, y: controlY },
      { x: targetX - 10, y: targetY },
      { x: targetX, y: targetY },
    ]
  } else {
    // Right-to-left: down → left → up
    const controlY = sourceY + baseOffset + offset

    return [
      { x: sourceX, y: sourceY },
      { x: sourceX - 10, y: sourceY },
      { x: sourceX - 10, y: controlY },
      { x: targetX + 10, y: controlY },
      { x: targetX + 10, y: targetY },
      { x: targetX, y: targetY },
    ]
  }
}

/**
 * Check if two shapes are adjacent (no other shapes between them horizontally)
 */
function areAdjacent(shape1, shape2, allShapes) {
  const minX = Math.min(shape1.x, shape2.x)
  const maxX = Math.max(shape1.x + shape1.width, shape2.x + shape2.width)

  // Check if there are any shapes between them
  for (const shape of allShapes) {
    if (shape.id === shape1.id || shape.id === shape2.id) continue
    if (shape.type !== 'board:ListBox') continue

    const shapeLeft = shape.x
    const shapeRight = shape.x + shape.width

    // If a shape is between them, they're not adjacent
    if (shapeLeft > minX && shapeRight < maxX) {
      return false
    }
  }

  return true
}

/**
 * Calculate all edge routes collectively to assign unique heights per edge
 * This should be called before creating connections to get properly indexed edges
 */
export function calculateAllEdgeRoutes(semanticPreferences, listShapes, allListShapesById) {
  // Create list of all edges with their indices
  const edgesWithIndices = semanticPreferences.map((pref, idx) => ({
    source: pref.source || pref[0],
    target: pref.target || pref[1],
    index: idx,
  }))

  // Sort edges by source position, then target position
  // This ensures consistent ordering for edge levels
  const listOrder = Object.keys(allListShapesById).sort((a, b) => {
    const shapeA = allListShapesById[a]
    const shapeB = allListShapesById[b]
    return shapeA.x - shapeB.x
  })

  const listIndexMap = {}
  listOrder.forEach((listId, idx) => {
    listIndexMap[listId] = idx
  })

  edgesWithIndices.sort((e1, e2) => {
    const sourceOrder1 = listIndexMap[e1.source] || 0
    const sourceOrder2 = listIndexMap[e2.source] || 0
    if (sourceOrder1 !== sourceOrder2) return sourceOrder1 - sourceOrder2

    const targetOrder1 = listIndexMap[e1.target] || 0
    const targetOrder2 = listIndexMap[e2.target] || 0
    return targetOrder1 - targetOrder2
  })

  // Calculate routes with unique levels per edge
  const routes = []
  const allShapes = Object.values(allListShapesById)

  edgesWithIndices.forEach((edgeInfo, level) => {
    const sourceShape = allListShapesById[edgeInfo.source]
    const targetShape = allListShapesById[edgeInfo.target]

    if (sourceShape && targetShape) {
      // Pass level instead of edgeIndex/totalEdges
      // This gives each unique edge a unique vertical offset
      const waypoints = calculateEdgeRoutingWithLevel(
        sourceShape,
        targetShape,
        level,
        edgesWithIndices.length,
        allShapes
      )

      routes[edgeInfo.index] = waypoints
    }
  })

  return routes
}

/**
 * Calculate routing using absolute edge level (not relative index within pair)
 */
function calculateEdgeRoutingWithLevel(sourceShape, targetShape, edgeLevel, totalEdges, allShapes) {
  if (!sourceShape || !targetShape) return null

  const isAdjacent = areAdjacent(sourceShape, targetShape, allShapes)
  const isLeftToRight = sourceShape.x < targetShape.x

  let sourceY, targetY, sourceX, targetX

  if (isAdjacent) {
    // Adjacent boxes: horizontal line
    if (isLeftToRight) {
      // Left-to-right: top-middle to top-middle
      sourceY = sourceShape.y + sourceShape.height / 2 - 30
      targetY = targetShape.y + targetShape.height / 2 - 30
      sourceX = sourceShape.x + sourceShape.width
      targetX = targetShape.x
    } else {
      // Right-to-left: bottom-middle to bottom-middle
      sourceY = sourceShape.y + sourceShape.height / 2 + 30
      targetY = targetShape.y + targetShape.height / 2 + 30
      sourceX = sourceShape.x
      targetX = targetShape.x + targetShape.width
    }

    return [
      { x: sourceX, y: sourceY },
      { x: targetX, y: targetY },
    ]
  }

  // Non-adjacent boxes with orthogonal routing
  if (isLeftToRight) {
    // Left-to-right: exit from top-middle of source, enter from top-middle of target
    sourceY = sourceShape.y + sourceShape.height / 2 - 30
    targetY = targetShape.y + targetShape.height / 2 - 30
    sourceX = sourceShape.x + sourceShape.width
    targetX = targetShape.x
  } else {
    // Right-to-left: exit from bottom-middle of source, enter from bottom-middle of target
    sourceY = sourceShape.y + sourceShape.height / 2 + 30
    targetY = targetShape.y + targetShape.height / 2 + 30
    sourceX = sourceShape.x
    targetX = targetShape.x + targetShape.width
  }

  // Distribute edges at different heights based on their level
  const baseOffset = 80 // distance from source box to start routing
  const minSeparation = 10 // 10px between consecutive edge levels
  const levelOffset = edgeLevel * minSeparation

  if (isLeftToRight) {
    // Left-to-right: go up, then right, then down
    const controlY = sourceY - baseOffset - levelOffset

    return [
      { x: sourceX, y: sourceY },
      { x: sourceX + 10, y: sourceY },
      { x: sourceX + 10, y: controlY },
      { x: targetX - 10, y: controlY },
      { x: targetX - 10, y: targetY },
      { x: targetX, y: targetY },
    ]
  } else {
    // Right-to-left: go down, then left, then up
    const controlY = sourceY + baseOffset + levelOffset

    return [
      { x: sourceX, y: sourceY },
      { x: sourceX - 10, y: sourceY },
      { x: sourceX - 10, y: controlY },
      { x: targetX + 10, y: controlY },
      { x: targetX + 10, y: targetY },
      { x: targetX, y: targetY },
    ]
  }
}
