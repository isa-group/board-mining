<script>
  import { onMount } from 'svelte'
  import { getBoardInfo, getHealthSummary } from '../api.js'
  import { selectedPeriod } from '../stores.js'

  let info = $state(null)
  let health = $state(null)
  let error = $state(null)

  onMount(async () => {
    try {
      ;[info, health] = await Promise.all([getBoardInfo(), getHealthSummary()])
    } catch (e) {
      error = e.message
    }
  })

  function fmt(val, isPercent = false) {
    if (val == null) return '–'
    if (isPercent) return `${(val * 100).toFixed(1)}%`
    if (typeof val === 'number') return val.toLocaleString()
    return String(val)
  }

  function scoreColor(val) {
    if (val == null) return 'var(--text-muted)'
    if (val >= 0.7) return 'var(--success)'
    if (val >= 0.4) return 'var(--warning)'
    return 'var(--danger)'
  }
</script>

<div class="overview">
  {#if error}
    <p class="error">{error}</p>
  {:else if !info}
    <p class="loading">Loading…</p>
  {:else}
    <section class="kpi-grid">
      <div class="kpi">
        <span class="kpi-label">Cards</span>
        <span class="kpi-value">{fmt(info.cards)}</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">Events</span>
        <span class="kpi-value">{fmt(info.events)}</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">Lists</span>
        <span class="kpi-value">{fmt(info.lists)}</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">Date range</span>
        <span class="kpi-value sm">{info.start?.slice(0,10) ?? '–'} → {info.ends?.slice(0,10) ?? '–'}</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">Completion rate</span>
        <span class="kpi-value" style="color:{scoreColor(health?.completion_rate)}">{fmt(health?.completion_rate, true)}</span>
      </div>
      <div class="kpi">
        <span class="kpi-label">Abandonment rate</span>
        <span class="kpi-value" style="color:{scoreColor(health ? 1 - health.abandonment_rate : null)}">{fmt(health?.abandonment_rate, true)}</span>
      </div>
    </section>

    {#if health}
      <section class="dims">
        <h2 class="section-title">Health dimensions</h2>
        <div class="dim-bars">
          {#each Object.entries(health).filter(([k]) => k.startsWith('dim_')) as [key, val]}
            <div class="dim-row">
              <span class="dim-label">{key.replace('dim_', '').replace(/_/g, ' ')}</span>
              <div class="bar-track">
                <div class="bar-fill" style="width:{(val * 100).toFixed(1)}%; background:{scoreColor(val)}"></div>
              </div>
              <span class="dim-score" style="color:{scoreColor(val)}">{(val * 100).toFixed(0)}</span>
            </div>
          {/each}
        </div>
      </section>
    {/if}
  {/if}
</div>

<style>
  .overview { display: flex; flex-direction: column; gap: 32px; flex: 1; overflow-y: auto; }

  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 16px;
  }

  .kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .kpi-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); }
  .kpi-value { font-size: 28px; font-weight: 700; line-height: 1; }
  .kpi-value.sm { font-size: 13px; line-height: 1.4; }

  .section-title { font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px; }

  .dim-bars { display: flex; flex-direction: column; gap: 12px; max-width: 540px; }

  .dim-row { display: grid; grid-template-columns: 200px 1fr 36px; align-items: center; gap: 12px; }

  .dim-label { font-size: 13px; text-transform: capitalize; }

  .bar-track { height: 6px; background: var(--surface-2); border-radius: 3px; overflow: hidden; }

  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }

  .dim-score { font-size: 12px; font-weight: 600; text-align: right; }

  .loading, .error { color: var(--text-muted); padding: 24px 0; }
  .error { color: var(--danger); }
</style>
