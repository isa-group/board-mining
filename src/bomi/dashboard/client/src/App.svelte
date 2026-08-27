<script>
  import { boardLoaded, boardInfo } from './stores.js'
  import DataPanel from './components/DataPanel.svelte'
  import Overview from './components/Overview.svelte'
  import Structure from './components/Structure.svelte'
  import Flow from './components/Flow.svelte'
  import Health from './components/Health.svelte'
  import Cards from './components/Cards.svelte'

  let activeTab = $state('overview')

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'structure',  label: 'Structure' },
    { id: 'flow',       label: 'Flow' },
    { id: 'health',     label: 'Health' },
    { id: 'cards',      label: 'Cards' },
  ]
</script>

<div class="shell">
  <header class="topbar">
    <div class="brand">
      <span class="logo">◈</span>
      <span class="name">bomi</span>
      <span class="subtitle">board dashboard</span>
    </div>
    <DataPanel />
  </header>

  {#if $boardLoaded}
    <nav class="tabs">
      {#each tabs as tab}
        <button
          class="tab-btn"
          class:active={activeTab === tab.id}
          onclick={() => activeTab = tab.id}
        >
          {tab.label}
        </button>
      {/each}
    </nav>

    <main class="content">
      {#if activeTab === 'overview'}   <Overview />   {/if}
      {#if activeTab === 'structure'}  <Structure />  {/if}
      {#if activeTab === 'flow'}       <Flow />       {/if}
      {#if activeTab === 'health'}     <Health />     {/if}
      {#if activeTab === 'cards'}      <Cards />      {/if}
    </main>
  {:else}
    <div class="empty-state">
      <p>Load a board using the panel above to begin.</p>
    </div>
  {/if}
</div>

<style>
  .shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  .topbar {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 0 24px;
    height: 56px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .brand {
    display: flex;
    align-items: baseline;
    gap: 8px;
    flex-shrink: 0;
  }

  .logo {
    font-size: 20px;
    color: var(--accent);
  }

  .name {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text);
  }

  .subtitle {
    font-size: 12px;
    color: var(--text-muted);
  }

  .tabs {
    display: flex;
    gap: 2px;
    padding: 0 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .tab-btn {
    padding: 10px 18px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    background: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    transition: color 0.15s, border-color 0.15s;
  }

  .tab-btn:hover { color: var(--text); }

  .tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  .content {
    flex: 1;
    min-height: 0;        /* allow flex child to shrink below content size */
    overflow: hidden;     /* no page-level scroll; each tab manages its own */
    display: flex;
    flex-direction: column;
    padding: 24px;
  }

  .empty-state {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 15px;
  }
</style>
