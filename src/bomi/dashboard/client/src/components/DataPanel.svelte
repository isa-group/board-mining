<script>
  import { uploadFile, loadFromId, getBoardInfo } from '../api.js'
  import { boardLoaded, boardInfo, loadError, resetSelections } from '../stores.js'

  let mode = $state('file')        // 'file' | 'id' | 'trello'
  let boardId = $state('')
  let loading = $state(false)
  let dragOver = $state(false)

  async function onBoardReady() {
    const info = await getBoardInfo()
    boardInfo.set(info)
    boardLoaded.set(true)
    loadError.set(null)
    resetSelections()
  }

  async function handleFile(file) {
    if (!file) return
    loading = true
    loadError.set(null)
    try {
      await uploadFile(file)
      await onBoardReady()
    } catch (e) {
      loadError.set(e.message)
    } finally {
      loading = false
    }
  }

  async function handleBoardId() {
    if (!boardId.trim()) return
    loading = true
    loadError.set(null)
    try {
      await loadFromId(boardId.trim())
      await onBoardReady()
    } catch (e) {
      loadError.set(e.message)
    } finally {
      loading = false
    }
  }

  function onFileInput(e) {
    handleFile(e.target.files?.[0])
  }

  function onDrop(e) {
    e.preventDefault()
    dragOver = false
    handleFile(e.dataTransfer.files?.[0])
  }
</script>

<div class="data-panel">
  <div class="mode-pills">
    <button class="pill" class:active={mode === 'file'} onclick={() => mode = 'file'}>File</button>
    <button class="pill" class:active={mode === 'id'}   onclick={() => mode = 'id'}>Board ID</button>
    <button class="pill disabled" disabled title="Coming soon">My Trello ↗</button>
  </div>

  {#if mode === 'file'}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="drop-zone"
      class:drag-over={dragOver}
      ondragover={(e) => { e.preventDefault(); dragOver = true }}
      ondragleave={() => dragOver = false}
      ondrop={onDrop}
    >
      <label class="upload-label">
        <input type="file" accept=".csv,.json" onchange={onFileInput} hidden />
        {#if loading}
          <span class="spinner"></span>
        {:else}
          <span>Drop CSV / JSON or <u>browse</u></span>
        {/if}
      </label>
    </div>

  {:else if mode === 'id'}
    <div class="id-row">
      <input
        type="text"
        placeholder="Trello board ID or URL"
        bind:value={boardId}
        onkeydown={(e) => e.key === 'Enter' && handleBoardId()}
      />
      <button class="load-btn" onclick={handleBoardId} disabled={loading || !boardId.trim()}>
        {#if loading}<span class="spinner sm"></span>{:else}Load{/if}
      </button>
    </div>
  {/if}

  {#if $loadError}
    <p class="error">{$loadError}</p>
  {/if}

  {#if $boardInfo}
    <span class="board-badge">{$boardInfo.cards ?? '–'} cards · {$boardInfo.events ?? '–'} events</span>
  {/if}
</div>

<style>
  .data-panel {
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 1;
  }

  .mode-pills {
    display: flex;
    gap: 4px;
    flex-shrink: 0;
  }

  .pill {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: var(--surface-2);
    color: var(--text-muted);
    border: 1px solid var(--border);
    transition: background 0.15s, color 0.15s;
  }

  .pill:hover:not(:disabled) { color: var(--text); }

  .pill.active {
    background: var(--accent-dim);
    color: var(--text);
    border-color: var(--accent);
  }

  .pill.disabled {
    opacity: 0.4;
    cursor: default;
  }

  .drop-zone {
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: 6px 16px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    font-size: 12px;
    color: var(--text-muted);
  }

  .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(108, 142, 245, 0.08);
    color: var(--accent);
  }

  .upload-label {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .id-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .id-row input { width: 220px; }

  .load-btn {
    padding: 7px 14px;
    border-radius: var(--radius);
    background: var(--accent);
    color: #fff;
    font-size: 13px;
    font-weight: 500;
    transition: opacity 0.15s;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .load-btn:disabled { opacity: 0.5; cursor: default; }

  .error {
    font-size: 12px;
    color: var(--danger);
    max-width: 260px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .board-badge {
    font-size: 11px;
    color: var(--text-muted);
    padding: 3px 8px;
    border: 1px solid var(--border);
    border-radius: 12px;
    white-space: nowrap;
  }

  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  .spinner.sm { width: 12px; height: 12px; }

  @keyframes spin { to { transform: rotate(360deg); } }
</style>
