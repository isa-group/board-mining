<script>
  import Diagram from 'diagram-js'
  import BoardRendererModule from '../lib/board-renderer.js'
  import { createBoardRules } from '../lib/board-rules.js'
  import { calculateLayout } from '../lib/board-layout.js'
  import { calculateAllEdgeRoutes } from '../lib/board-edge-router.js'
  import { selectedList, selectedTransition } from '../stores.js'
  import '../assets/board-renderer.css'

  let { boardData = null, readOnly = true, onModelChange = null } = $props()

  let container = $state(null)
  let diagram = null
  let spacePressed = $state(false)

  // Watch for boardData changes and re-render
  $effect(() => {
    if (boardData && container) {
      const cleanup = initializeDiagram()
      return cleanup
    }
  })

  function initializeDiagram() {
    if (!container || !boardData) return

    let isPanning = false
    let panStart = { x: 0, y: 0 }
    const cleanupHandlers = []

    try {
      // Clear any existing content in the container
      if (container) {
        container.innerHTML = ''
      }

      // Initialize diagram-js instance
      diagram = new Diagram({
        container,
        canvas: { viewbox: { x: 0, y: 0, width: 1400, height: 900 } },
        modules: [BoardRendererModule, createBoardRules()],
        keyboard: !readOnly,
        selection: !readOnly,
        move: !readOnly,
        resize: !readOnly,
        bendpoints: !readOnly,
        zoomScroll: {
          enabled: true,
          scale: 0.75,
        },
      })

      console.log('Diagram initialized. Container:', container)

      // Wait a moment for diagram-js to create the djs-container, then move it if needed
      setTimeout(() => {
        const djsContainer = document.querySelector('.djs-container')
        if (djsContainer && container && !container.contains(djsContainer)) {
          console.log('Moving djs-container to correct location')
          container.appendChild(djsContainer)
        }
        console.log('DJS container element:', container?.querySelector('.djs-container'))
      }, 100)

      // Add space+drag and middle-click panning functionality
      const canvas = diagram.get('canvas')
      if (canvas && canvas._container) {
        // Detect space key for panning
        const keydownHandler = (e) => {
          if (e.code === 'Space') {
            spacePressed = true
            e.preventDefault()
          }
        }
        const keyupHandler = (e) => {
          if (e.code === 'Space') {
            spacePressed = false
          }
        }

        const mousedownHandler = (e) => {
          // Middle-click or space+left-click to pan
          if (e.button === 1 || (e.button === 0 && spacePressed)) {
            isPanning = true
            panStart = { x: e.clientX, y: e.clientY }
            e.preventDefault()
          }
        }

        const mousemoveHandler = (e) => {
          if (isPanning) {
            const deltaX = e.clientX - panStart.x
            const deltaY = e.clientY - panStart.y

            // Get current viewbox and pan by moving it
            const currentViewbox = canvas.viewbox()
            canvas.viewbox({
              x: currentViewbox.x - deltaX,
              y: currentViewbox.y - deltaY,
              width: currentViewbox.width,
              height: currentViewbox.height,
            })

            panStart = { x: e.clientX, y: e.clientY }
          }
        }

        const mouseupHandler = () => {
          isPanning = false
        }

        const wheelHandler = (e) => {
          // Space + mouse wheel for zoom
          if (spacePressed) {
            e.preventDefault()
            const currentViewbox = canvas.viewbox()
            const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9  // Scroll down = zoom out, up = zoom in
            const newWidth = currentViewbox.width * zoomFactor
            const newHeight = currentViewbox.height * zoomFactor

            // Zoom towards mouse position
            const rect = canvas._container.getBoundingClientRect()
            const mouseX = e.clientX - rect.left
            const mouseY = e.clientY - rect.top
            const percentX = mouseX / rect.width
            const percentY = mouseY / rect.height

            canvas.viewbox({
              x: currentViewbox.x + (currentViewbox.width - newWidth) * percentX,
              y: currentViewbox.y + (currentViewbox.height - newHeight) * percentY,
              width: newWidth,
              height: newHeight,
            })
          }
        }

        document.addEventListener('keydown', keydownHandler)
        document.addEventListener('keyup', keyupHandler)
        canvas._container.addEventListener('mousedown', mousedownHandler)
        canvas._container.addEventListener('wheel', wheelHandler, { passive: false })
        document.addEventListener('mousemove', mousemoveHandler)
        document.addEventListener('mouseup', mouseupHandler)

        // Store cleanup functions
        cleanupHandlers.push(() => {
          document.removeEventListener('keydown', keydownHandler)
          document.removeEventListener('keyup', keyupHandler)
          canvas._container?.removeEventListener('mousedown', mousedownHandler)
          canvas._container?.removeEventListener('wheel', wheelHandler)
          document.removeEventListener('mousemove', mousemoveHandler)
          document.removeEventListener('mouseup', mouseupHandler)
        })
      }

      // Import board design into diagram
      try {
        importBoardDesign(boardData)
      } catch (err) {
        console.error('Error importing board design:', err)
      }

      // Setup event listeners for filter integration
      const eventBus = diagram.get('eventBus')

      eventBus.on('element.click', (e) => {
        if (e.element && e.element.type === 'board:ListBox') {
          const listId = e.element.businessObject?.listId
          if (listId) {
            selectedList.set(listId)
          }
        } else if (e.element && e.element.type === 'board:SemanticPreference') {
          const source = e.element.source?.businessObject?.name
          const target = e.element.target?.businessObject?.name
          if (source && target) {
            selectedTransition.set({ source, target })
          }
        }
      })

      // Setup model change notifications (for editor mode)
      if (!readOnly) {
        eventBus.on('element.changed', () => {
          onModelChange?.(exportBoardDesign())
        })
        eventBus.on('connection.added', () => {
          onModelChange?.(exportBoardDesign())
        })
        eventBus.on('element.removed', () => {
          onModelChange?.(exportBoardDesign())
        })
      }
    } catch (err) {
      console.error('Error initializing board design editor:', err)
    }

    // Setup cleanup for when component unmounts
    return () => {
      cleanupHandlers.forEach((cleanup) => cleanup())
      if (diagram) {
        try {
          diagram.destroy()
        } catch (err) {
          console.error('Error destroying diagram:', err)
        }
      }
    }
  }

  function importBoardDesign(boardModel) {
    if (!diagram || !boardModel) return

    const elementFactory = diagram.get('elementFactory')
    const canvas = diagram.get('canvas')
    const calculatedLayout = calculateLayout(boardModel)

    try {
      // Create CardFlow containers (one per connected component)
      // Use componentOrder from layout to ensure correct ordering
      const componentOrder = calculatedLayout.componentOrder || [boardModel.card_flow[0]] // Fallback for compatibility

      componentOrder.forEach((component, flatIdx) => {
        const containerPos = calculatedLayout.containers[flatIdx]
        if (!containerPos) return // Skip if position not calculated

        const cardFlowShape = elementFactory.createShape({
          type: 'board:CardFlow',
          id: `cardflow-${flatIdx}`,
          businessObject: { componentId: flatIdx.toString() },
          x: containerPos.x,
          y: containerPos.y,
          width: containerPos.width,
          height: containerPos.height,
        })
        canvas.addShape(cardFlowShape)

        // Create ListBox shapes inside CardFlow in the order from layout
        component.forEach((listId) => {
          const listPos = calculatedLayout.lists[listId]
          if (!listPos) return // Skip if layout not calculated

          const listShape = elementFactory.createShape({
            type: 'board:ListBox',
            id: `list-${listId}`,
            parent: cardFlowShape,
            x: listPos.x,
            y: listPos.y,
            width: 140,
            height: 80,
            businessObject: {
              listId,
              name: listId,
              isCreateList: boardModel.card_create_lists?.includes(listId) || false,
              isCloseList: boardModel.card_close_lists?.includes(listId) || false,
              isUseList: boardModel.card_use_lists?.includes(listId) || false,
              trafficVolume: 0,
            },
          })
          canvas.addShape(listShape)
        })
      })

      // Build a map of all list shapes for edge routing
      // First, find all shapes in the diagram (not just root-level)
      const allListShapesById = {}
      const allShapes = []

      function collectShapes(element) {
        if (element.type === 'board:ListBox') {
          allListShapesById[element.businessObject.listId] = element
          allShapes.push(element)
        }
        if (element.children) {
          element.children.forEach(collectShapes)
        }
      }

      const rootElement = canvas.getRootElement()
      if (rootElement.children) {
        rootElement.children.forEach(collectShapes)
      }

      console.log('List shapes for routing:', Object.keys(allListShapesById))
      console.log('Semantic preferences to route:', boardModel.semantic_precedence?.length || 0)
      console.log('Root element:', rootElement)
      console.log('Root element children:', rootElement.children?.length)

      // Calculate edge routes with orthogonal routing
      const edgeRoutes = calculateAllEdgeRoutes(
        boardModel.semantic_precedence || [],
        allShapes,
        allListShapesById
      )

      console.log('Edge routes calculated:', edgeRoutes)

      // Create SemanticPreference connections with pre-calculated waypoints
      boardModel.semantic_precedence?.forEach((preference, edgeIdx) => {
        const sourceListId = preference.source || preference[0]
        const targetListId = preference.target || preference[1]
        const volume = preference.volume || preference[2] || 0

        const sourceShape = allListShapesById[sourceListId]
        const targetShape = allListShapesById[targetListId]

        console.log(`Edge ${edgeIdx}: ${sourceListId} → ${targetListId}`, {
          sourceShape: !!sourceShape,
          targetShape: !!targetShape,
          waypoints: edgeRoutes[edgeIdx],
        })

        if (sourceShape && targetShape) {
          const waypoints = edgeRoutes[edgeIdx] || [
            { x: sourceShape.x + sourceShape.width / 2, y: sourceShape.y + sourceShape.height / 2 },
            { x: targetShape.x + targetShape.width / 2, y: targetShape.y + targetShape.height / 2 },
          ]

          const connection = elementFactory.createConnection({
            type: 'board:SemanticPreference',
            source: sourceShape,
            target: targetShape,
            waypoints,
            businessObject: {
              sourceList: sourceListId,
              targetList: targetListId,
              volume: volume || 0,
            },
          })
          canvas.addConnection(connection)
        }
      })
    } catch (err) {
      console.error('Error importing board design:', err)
    }
  }

  function exportBoardDesign() {
    if (!diagram) return null

    const canvas = diagram.get('canvas')
    const rootElement = canvas.getRootElement()

    // Extract shapes and connections from diagram
    const lists = []
    const semanticPreferences = []
    const cardFlowComponents = []

    rootElement.children?.forEach((element) => {
      if (element.type === 'board:CardFlow') {
        const componentLists = element.children
          ?.filter((child) => child.type === 'board:ListBox')
          .map((child) => child.businessObject?.listId) || []
        cardFlowComponents.push(componentLists)
      }
    })

    rootElement.children?.forEach((element) => {
      if (element.type === 'board:ListBox') {
        lists.push({
          id: element.businessObject?.listId,
          name: element.businessObject?.name,
          trafficVolume: element.businessObject?.trafficVolume || 0,
        })
      }
    })

    rootElement.outgoing?.forEach((connection) => {
      if (connection.type === 'board:SemanticPreference') {
        semanticPreferences.push({
          source: connection.source?.businessObject?.listId,
          target: connection.target?.businessObject?.listId,
          volume: connection.businessObject?.volume || 0,
        })
      }
    })

    return {
      lists,
      card_flow: cardFlowComponents,
      semantic_precedence: semanticPreferences,
    }
  }
</script>

<div bind:this={container} class="board-editor"></div>

<style>
  .board-editor {
    width: 100%;
    height: 100%;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    position: relative;
    overflow: hidden;
  }

  :global(.board-editor .djs-container) {
    background: var(--surface);
    height: 100% !important;
  }

  :global(.board-editor .djs-viewport) {
    height: 100% !important;
    overflow: visible !important;
  }

  :global(.board-editor .djs-canvas) {
    background: var(--surface);
    height: 100% !important;
  }
</style>
