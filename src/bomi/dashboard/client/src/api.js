const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const body = await res.text()
    throw new Error(body || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Board loading ─────────────────────────────────────────────────────────────

export async function uploadFile(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/board/upload', { method: 'POST', body: form })
}

export async function loadFromId(boardId) {
  return request('/board/from-id', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ board_id: boardId }),
  })
}

export async function getBoardInfo() {
  return request('/board/info')
}

// ── Structure ─────────────────────────────────────────────────────────────────

export async function getListEvolution() {
  return request('/structure/list-evolution')
}

export async function getRedesigns(thresholdDays = 1, lType = null, thresholdLEvents = 0) {
  const params = new URLSearchParams()
  params.set('threshold_days', thresholdDays)
  if (lType) params.set('l_type', Array.isArray(lType) ? lType.join(',') : lType)
  params.set('threshold_l_events', thresholdLEvents)
  return request(`/structure/redesigns?${params}`)
}

export async function getEventsByList(startDate = null, endDate = null) {
  const params = new URLSearchParams()
  if (startDate) params.set('start_date', startDate.toISOString().split('T')[0])
  if (endDate) params.set('end_date', endDate.toISOString().split('T')[0])
  const qs = params.toString()
  return request(`/structure/events-by-list${qs ? '?' + qs : ''}`)
}

// ── Flow ──────────────────────────────────────────────────────────────────────

export async function getTransitionMatrix(startDate = null, endDate = null, sortBy = 'net_flow') {
  const params = new URLSearchParams()
  if (startDate) params.set('start_date', startDate.toISOString().split('T')[0])
  if (endDate) params.set('end_date', endDate.toISOString().split('T')[0])
  params.set('sort_by', sortBy)
  const qs = params.toString()
  return request(`/flow/transition-matrix${qs ? '?' + qs : ''}`)
}

export async function getConnectedLists() {
  return request('/flow/connected-lists')
}

export async function getSemanticPrecedence() {
  return request('/flow/semantic-precedence')
}

export async function getBoardDesign(
  startDate = null,
  endDate = null,
  cfThreshold = 0,
  ccThreshold = 0,
  cxThreshold = 0,
  cuThreshold = 0,
  spThreshold = 0
) {
  const params = new URLSearchParams()
  if (startDate) params.set('start_date', startDate.toISOString().split('T')[0])
  if (endDate) params.set('end_date', endDate.toISOString().split('T')[0])
  params.set('cf_threshold', cfThreshold)
  params.set('cc_threshold', ccThreshold)
  params.set('cx_threshold', cxThreshold)
  params.set('cu_threshold', cuThreshold)
  params.set('sp_threshold', spThreshold)
  const qs = params.toString()
  return request(`/flow/board-design?${qs}`)
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealthSummary() {
  return request('/health/summary')
}

export async function getHealthDimensions() {
  return request('/health/dimensions')
}

export async function getHealthEvolution(windowDays = 30, stepDays = 7, indicators = null) {
  const params = new URLSearchParams({ window_days: windowDays, step_days: stepDays })
  if (indicators) params.set('indicators', indicators.join(','))
  return request(`/health/evolution?${params}`)
}

// ── Cards ─────────────────────────────────────────────────────────────────────

export async function getCardIndicators(sourceList = null, targetList = null, listId = null) {
  const params = new URLSearchParams()
  if (sourceList) params.set('source_list', sourceList)
  if (targetList) params.set('target_list', targetList)
  if (listId) params.set('list_id', listId)
  const qs = params.toString()
  return request(`/cards/indicators${qs ? '?' + qs : ''}`)
}

export async function getCardTimeline(cardId) {
  return request(`/cards/timeline/${encodeURIComponent(cardId)}`)
}
