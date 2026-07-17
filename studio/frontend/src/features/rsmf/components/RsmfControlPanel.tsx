/**
 * RSMF Control Panel — configuration UI for model & training parameters.
 */
import { useRsmfStore } from '../stores/rsmf-store';

export function RsmfControlPanel() {
  const {
    modelConfig, trainConfig, initResult, isTraining,
    lossHistory, coherenceHistory, budgetResult,
    setModelConfig, setTrainConfig, initializeModel, computeBudget,
  } = useRsmfStore();

  return (
    <div className="flex flex-col gap-4 p-4 text-sm">
      {/* Model Config */}
      <section className="border border-white/10 rounded-lg p-3 bg-white/5">
        <h3 className="text-xs uppercase tracking-wider text-white/60 mb-2">Model Architecture</h3>
        <div className="grid grid-cols-2 gap-2">
          <label className="flex flex-col gap-1">
            <span className="text-white/50 text-xs">Architecture</span>
            <select
              value={modelConfig.architecture}
              onChange={(e) => setModelConfig({ architecture: e.target.value as 'dense' | 'moe' })}
              className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white"
            >
              <option value="dense">Dense</option>
              <option value="moe">MoE</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-white/50 text-xs">Hidden Dim</span>
            <input type="number" value={modelConfig.hidden_dim}
              onChange={(e) => setModelConfig({ hidden_dim: +e.target.value })}
              className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-white/50 text-xs">Layers</span>
            <input type="number" value={modelConfig.num_layers}
              onChange={(e) => setModelConfig({ num_layers: +e.target.value })}
              className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-white/50 text-xs">Heads</span>
            <input type="number" value={modelConfig.num_heads}
              onChange={(e) => setModelConfig({ num_heads: +e.target.value })}
              className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
          </label>
          {modelConfig.architecture === 'moe' && (
            <>
              <label className="flex flex-col gap-1">
                <span className="text-white/50 text-xs">Experts</span>
                <input type="number" value={modelConfig.num_experts ?? 8}
                  onChange={(e) => setModelConfig({ num_experts: +e.target.value })}
                  className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-white/50 text-xs">Active Experts</span>
                <input type="number" value={modelConfig.active_experts ?? 2}
                  onChange={(e) => setModelConfig({ active_experts: +e.target.value })}
                  className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
              </label>
            </>
          )}
        </div>
      </section>

      {/* Training Config */}
      <section className="border border-white/10 rounded-lg p-3 bg-white/5">
        <h3 className="text-xs uppercase tracking-wider text-white/60 mb-2">RSMF Parameters</h3>
        <div className="grid grid-cols-2 gap-2">
          <label className="flex flex-col gap-1">
            <span className="text-white/50 text-xs">Top-K (spectral rank)</span>
            <input type="number" value={trainConfig.top_k}
              onChange={(e) => setTrainConfig({ top_k: +e.target.value })}
              className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-white/50 text-xs">λ Spectral</span>
            <input type="number" step="0.001" value={trainConfig.lambda_spectral}
              onChange={(e) => setTrainConfig({ lambda_spectral: +e.target.value })}
              className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-white/50 text-xs">μ Coupling</span>
            <input type="number" step="0.01" value={trainConfig.mu_coupling}
              onChange={(e) => setTrainConfig({ mu_coupling: +e.target.value })}
              className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-white/50 text-xs">Learning Rate</span>
            <input type="number" step="0.0001" value={trainConfig.learning_rate}
              onChange={(e) => setTrainConfig({ learning_rate: +e.target.value })}
              className="bg-black/30 border border-white/10 rounded px-2 py-1 text-white" />
          </label>
        </div>
      </section>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={() => initializeModel()}
          disabled={isTraining}
          className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded px-3 py-2 font-medium transition-colors"
        >
          {initResult ? 'Reinitialize' : 'Initialize Model'}
        </button>
        <button
          onClick={() => computeBudget(4, 24)}
          className="flex-1 bg-purple-600 hover:bg-purple-500 text-white rounded px-3 py-2 font-medium transition-colors"
        >
          Check VRAM Budget
        </button>
      </div>

      {/* Stats */}
      {initResult && (
        <section className="border border-white/10 rounded-lg p-3 bg-white/5">
          <h3 className="text-xs uppercase tracking-wider text-white/60 mb-2">Model Stats</h3>
          <div className="grid grid-cols-2 gap-1 text-xs">
            <div className="text-white/50">Parameters:</div>
            <div className="text-white">{(initResult.total_parameters / 1e6).toFixed(1)}M</div>
            <div className="text-white/50">Compression:</div>
            <div className="text-white">{initResult.compression_ratio.toFixed(1)}×</div>
            <div className="text-white/50">VRAM/stratum:</div>
            <div className="text-white">{(initResult.vram_per_stratum_bytes / 1024).toFixed(0)} KB</div>
            <div className="text-white/50">Steps:</div>
            <div className="text-white">{lossHistory.length}</div>
            {lossHistory.length > 0 && (
              <>
                <div className="text-white/50">Last Loss:</div>
                <div className="text-white">{lossHistory[lossHistory.length - 1].toFixed(6)}</div>
              </>
            )}
          </div>
        </section>
      )}

      {/* Budget Result */}
      {budgetResult && (
        <section className={`border rounded-lg p-3 ${budgetResult.feasible ? 'border-green-500/30 bg-green-500/5' : 'border-red-500/30 bg-red-500/5'}`}>
          <h3 className="text-xs uppercase tracking-wider text-white/60 mb-2">VRAM Budget (4×24GB)</h3>
          <div className="text-xs">
            <span className={budgetResult.feasible ? 'text-green-400' : 'text-red-400'}>
              {budgetResult.feasible ? '✓ FEASIBLE' : '✗ NOT FEASIBLE'}
            </span>
            {budgetResult.allocations.map((a) => (
              <div key={a.device_id} className="mt-1 text-white/60">
                GPU{a.device_id}: k={a.recommended_top_k}, {(a.headroom_bytes / 1024 / 1024).toFixed(0)}MB free
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
