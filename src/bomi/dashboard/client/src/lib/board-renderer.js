/**
 * Board Design Custom Renderer
 *
 * Implements custom rendering for board design elements:
 * - ListBox: list boxes with role icons
 * - CardFlow: component containers
 * - SemanticPreference: dashed arrows with volume labels
 */

import BaseRenderer from 'diagram-js/lib/draw/BaseRenderer'

const HIGH_PRIORITY = 1500

// Helper to create SVG elements with proper namespace
function createSVGElement(tag, attrs = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag)
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, value)
  }
  return element
}

export class BoardRenderer extends BaseRenderer {
  constructor(eventBus) {
    super(eventBus, HIGH_PRIORITY)
  }

  canRender(element) {
    // Don't render label targets; handle text labels separately
    return !element.labelTarget
  }

  drawShape(parentGfx, element) {
    if (element.type === 'board:ListBox') {
      return this._drawListBox(parentGfx, element)
    }
    if (element.type === 'board:CardFlow') {
      return this._drawCardFlow(parentGfx, element)
    }

    return null
  }

  drawConnection(parentGfx, connection) {
    if (connection.type === 'board:SemanticPreference') {
      return this._drawSemanticPreference(parentGfx, connection)
    }

    return null
  }

  _drawListBox(parentGfx, element) {
    const { isCreateList, isCloseList, isUseList, name, listId } = element.businessObject || {}
    const displayName = name || listId || 'List'

    // Main rectangle for list box
    const rect = createSVGElement('rect', {
      x: 0,
      y: 0,
      width: element.width,
      height: element.height,
      rx: 4,
      class: 'board-list-box',
    })
    parentGfx.appendChild(rect)

    // Label with list name - wrapped text in the center of the box
    const text = createSVGElement('text', {
      x: element.width / 2,
      y: element.height / 2 - 10,
      'text-anchor': 'middle',
      'font-size': '11px',
      'font-weight': 'normal',
      fill: '#fff',
      'pointer-events': 'none',
    })

    // Split long names into multiple lines (fit to box width)
    const maxCharsPerLine = 20 // allows ~140px width with 11px font
    const words = displayName.split(' ')
    let currentLine = ''
    let lineNumber = 0

    words.forEach((word, idx) => {
      const testLine = currentLine + (currentLine ? ' ' : '') + word
      if (testLine.length > maxCharsPerLine && currentLine) {
        // Create tspan for current line
        const tspan = createSVGElement('tspan', {
          x: element.width / 2,
          dy: lineNumber === 0 ? '0' : '13',
        })
        tspan.textContent = currentLine
        text.appendChild(tspan)
        currentLine = word
        lineNumber++
      } else {
        currentLine = testLine
      }
    })

    // Add final line
    if (currentLine) {
      const tspan = createSVGElement('tspan', {
        x: element.width / 2,
        dy: lineNumber === 0 ? '0' : '14',
      })
      tspan.textContent = currentLine
      text.appendChild(tspan)
    }

    parentGfx.appendChild(text)

    // Role icons (bottom area)
    let iconX = element.width - 18
    const iconY = element.height - 20
    const iconSize = 14

    if (isCreateList) {
      this._drawIcon(parentGfx, iconX, iconY, '⊕', 'create', iconSize)
    }
    if (isUseList) {
      this._drawIcon(parentGfx, iconX - 16, iconY, '↻', 'use', iconSize)
    }
    if (isCloseList) {
      this._drawIcon(parentGfx, iconX - 32, iconY, '⊠', 'close', iconSize)
    }

    return rect
  }

  _drawCardFlow(parentGfx, element) {
    // Container box with dashed border
    const rect = createSVGElement('rect', {
      x: 0,
      y: 0,
      width: element.width,
      height: element.height,
      rx: 8,
      class: 'board-card-flow',
    })
    parentGfx.appendChild(rect)

    return rect
  }

  _drawSemanticPreference(parentGfx, connection) {
    const { volume } = connection.businessObject || {}

    // Draw path along waypoints with dashed style
    const waypoints = connection.waypoints || []

    if (waypoints.length < 2) {
      return null
    }

    // Create path data from waypoints
    let pathData = `M ${waypoints[0].x} ${waypoints[0].y}`
    for (let i = 1; i < waypoints.length; i++) {
      pathData += ` L ${waypoints[i].x} ${waypoints[i].y}`
    }

    const path = createSVGElement('path', {
      d: pathData,
      class: 'board-semantic-preference',
      'stroke-width': Math.max(1, (volume || 0) / 10),
    })
    parentGfx.appendChild(path)

    // Add arrow marker
    const defs = parentGfx.ownerSVGElement?.querySelector('defs') || parentGfx.ownerSVGElement?.appendChild(
      createSVGElement('defs', {})
    )
    this._ensureArrowMarker(defs)

    path.setAttribute('marker-end', 'url(#board-arrow-marker)')

    // Volume label at midpoint
    if (volume && volume > 0) {
      const midIdx = Math.floor(waypoints.length / 2)
      const { x, y } = waypoints[midIdx]

      const volumeText = createSVGElement('text', {
        x,
        y: y - 8,
        'text-anchor': 'middle',
        class: 'board-volume-label',
      })
      volumeText.textContent = volume.toString()
      parentGfx.appendChild(volumeText)
    }

    return path
  }

  _drawIcon(parentGfx, x, y, icon, type, size) {
    const text = createSVGElement('text', {
      x,
      y: y + size,
      'text-anchor': 'middle',
      'font-size': size,
      'font-weight': 'bold',
      class: `board-icon ${type}`,
    })
    text.textContent = icon
    parentGfx.appendChild(text)
  }

  _ensureArrowMarker(defs) {
    // Only create if it doesn't exist
    if (defs.querySelector('#board-arrow-marker')) {
      return
    }

    const marker = createSVGElement('marker', {
      id: 'board-arrow-marker',
      markerWidth: 10,
      markerHeight: 10,
      refX: 10,
      refY: 3,
      orient: 'auto',
    })

    const path = createSVGElement('path', {
      d: 'M0,0 L10,3 L0,6 Z',
      fill: 'var(--text-muted)',
    })
    marker.appendChild(path)
    defs.appendChild(marker)
  }
}

export default {
  __init__: ['boardRenderer'],
  boardRenderer: ['type', BoardRenderer],
}
