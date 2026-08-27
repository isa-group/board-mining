import { writable } from 'svelte/store'

// Board loading state
export const boardLoaded = writable(false)
export const boardInfo = writable(null)
export const loadError = writable(null)

// Cross-chart selection state — the heart of the linked interactivity
export const selectedPeriod     = writable(null)  // [Date, Date] or null
export const selectedList       = writable(null)  // list name (from Structure Gantt)
export const selectedCard       = writable(null)  // card_id string or null
export const selectedTransition = writable(null)  // { source, target } or null (from Flow matrix/DFG)

// Health evolution cache (avoids re-fetching on tab switch)
export const healthEvolutionCache = writable(null)

// Redesign detection parameters (persist across tabs)
export const redesignThresholdDays = writable(1)
export const redesignThresholdLEvents = writable(0)
export const redesignPeriods = writable([])  // cached redesigns from last Structure query

// Flow tab configuration (persist across tabs)
export const flowShowPerRedesign = writable(false)
export const flowSelectedRedesign = writable(null)

// Clear all selections (called when a new board is loaded)
export function resetSelections() {
  selectedPeriod.set(null)
  selectedList.set(null)
  selectedCard.set(null)
  selectedTransition.set(null)
  healthEvolutionCache.set(null)
  redesignThresholdDays.set(1)
  redesignThresholdLEvents.set(0)
  redesignPeriods.set([])
  flowShowPerRedesign.set(false)
  flowSelectedRedesign.set(null)
}
