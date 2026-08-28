/**
 * Board Design Layout Engine
 *
 * Calculates positions for left-to-right topological layout.
 * All lists flow left-to-right based on semantic precedence order.
 * Connected components are shown as dashed container boxes.
 */

export function calculateLayout(boardModel) {
  // 1. Order components by size (largest first)
  const componentsWithIndices = boardModel.card_flow.map((component, idx) => ({
    lists: component,
    originalIdx: idx,
    size: component.length,
  }))
  componentsWithIndices.sort((a, b) => b.size - a.size)

  // 2. For each component, order lists by semantic precedence
  const orderedComponentLists = componentsWithIndices.map((comp) => {
    return topologicalSortWithin(boardModel.semantic_precedence, comp.lists)
  })

  // 3. Flatten the component-ordered lists and map original component indices
  const listToComponentIndex = {}
  const flatOrderedLists = []
  orderedComponentLists.forEach((componentLists, flatIdx) => {
    componentLists.forEach((listId) => {
      flatOrderedLists.push(listId)
      listToComponentIndex[listId] = flatIdx // Maps to flattened component order
    })
  })

  // 4. Assign X positions (left-to-right in a single row) with gaps between components
  const listPositions = {}
  const listWidth = 140
  const listHeight = 80
  const horizontalSpacing = 40  // Increased from 20 to 30 for better arrow spacing
  const componentMargin = 40  // Margin between components
  const startX = 100
  const startY = 150 // increased to leave room for edges above

  let currentX = startX
  orderedComponentLists.forEach((componentLists, componentIdx) => {
    // Position lists for this component
    componentLists.forEach((listId) => {
      listPositions[listId] = {
        x: currentX,
        y: startY,
      }
      currentX += listWidth + horizontalSpacing
    })

    // Add margin between components (but not after the last one)
    if (componentIdx < orderedComponentLists.length - 1) {
      currentX += componentMargin
    }
  })

  // 5. Calculate container bounds for each connected component
  const containers = {}
  const padding = 15

  orderedComponentLists.forEach((componentLists, flatComponentIdx) => {
    const componentListIds = componentLists.filter((listId) => flatOrderedLists.includes(listId))

    if (componentListIds.length === 0) return

    const positions = componentListIds.map((id) => listPositions[id])
    const minX = Math.min(...positions.map((p) => p.x))
    const maxX = Math.max(...positions.map((p) => p.x + listWidth))
    const minY = Math.min(...positions.map((p) => p.y))
    const maxY = Math.max(...positions.map((p) => p.y + listHeight))

    // Store with flattened index to match the import order in BoardDesignEditor
    containers[flatComponentIdx] = {
      x: minX - padding,
      y: minY - padding,
      width: maxX - minX + padding * 2,
      height: maxY - minY + padding * 2,
    }
  })

  return {
    lists: listPositions,
    containers,
    componentOrder: orderedComponentLists,
  }
}

/**
 * Topological sort of lists based on semantic precedence
 * Returns lists ordered from sources to sinks
 */
function topologicalSort(semanticPreferences, allLists) {
  const inDegree = {}
  const adjList = {}

  // Initialize
  allLists.forEach((listId) => {
    inDegree[listId] = 0
    adjList[listId] = []
  })

  // Build adjacency list
  semanticPreferences.forEach((pref) => {
    // Handle both array format [source, target] and object format {source, target}
    const source = pref.source || pref[0]
    const target = pref.target || pref[1]

    if (source && target && adjList[source] && adjList[target]) {
      adjList[source].push(target)
      inDegree[target]++
    }
  })

  // Kahn's algorithm for topological sort
  const queue = allLists.filter((id) => inDegree[id] === 0)
  const sorted = []

  while (queue.length > 0) {
    const node = queue.shift()
    sorted.push(node)

    adjList[node].forEach((neighbor) => {
      inDegree[neighbor]--
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor)
      }
    })
  }

  // Add any remaining nodes (disconnected components)
  allLists.forEach((id) => {
    if (!sorted.includes(id)) {
      sorted.push(id)
    }
  })

  return sorted
}

/**
 * Topological sort of lists within a single component
 * Only considers edges between lists in the given component
 */
function topologicalSortWithin(semanticPreferences, componentLists) {
  const listSet = new Set(componentLists)
  const inDegree = {}
  const adjList = {}

  // Initialize only with lists in this component
  componentLists.forEach((listId) => {
    inDegree[listId] = 0
    adjList[listId] = []
  })

  // Build adjacency list using only edges within component
  semanticPreferences.forEach((pref) => {
    const source = pref.source || pref[0]
    const target = pref.target || pref[1]

    if (source && target && listSet.has(source) && listSet.has(target)) {
      adjList[source].push(target)
      inDegree[target]++
    }
  })

  // Kahn's algorithm
  const queue = componentLists.filter((id) => inDegree[id] === 0)
  const sorted = []

  while (queue.length > 0) {
    const node = queue.shift()
    sorted.push(node)

    adjList[node].forEach((neighbor) => {
      inDegree[neighbor]--
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor)
      }
    })
  }

  // Add remaining nodes (disconnected within component)
  componentLists.forEach((id) => {
    if (!sorted.includes(id)) {
      sorted.push(id)
    }
  })

  return sorted
}

/**
 * Calculate list box height proportional to traffic volume
 * Min 40px, max 120px
 */
function calculateListHeight(trafficVolume) {
  return Math.max(40, Math.min(120, trafficVolume * 2))
}
