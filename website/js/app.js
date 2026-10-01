/* ============================================================================
   GreenPEFT — Interactive Application Engine & Behaviors
   ============================================================================ */

(function () {
  'use strict';

  // Check prefers-reduced-motion
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Global Engine Reference
  const Engine = window.GPEngine;
  const Charts = window.GPCharts;

  /* --------------------------------------------------------------------------
     1. Recommender State & Defaults
     ----------------------------------------------------------------------- */
  const state = {
    vram: 16,
    accuracy: 0.90,
    carbon: 0.25,
    runtimeMinutes: 60,
    profile: 'balanced',
    backbone: '',
    method: ''
  };

  const DEFAULTS = {
    vram: 16,
    accuracy: 0.90,
    carbon: 0.25,
    runtimeMinutes: 60,
    profile: 'balanced',
    backbone: '',
    method: ''
  };

  // Human-readable method names
  const METHOD_NAMES = {
    full_ft: 'Full fine-tuning',
    lora: 'LoRA',
    qlora: 'QLoRA',
    lora_fa: 'LoRA-FA',
    lisa: 'LISA'
  };

  const BACKBONE_NAMES = {
    nano: 'SmolLM2-135M (0.1B)',
    micro: 'SmolLM2-360M (0.4B)',
    tiny: 'Qwen2.5-0.5B (Tiny)',
    tiny_q3: 'Qwen3-0.6B',
    small: 'TinyLlama-1.1B (Small)',
    small_px: 'Pythia-1.4B',
    medium: 'Qwen2.5-1.5B (Medium)',
    medium_sl: 'SmolLM2-1.7B',
    medium_q3: 'Qwen3-1.7B',
    large: 'Qwen2.5-3B (Large)',
    large_phi: 'Phi-2-2.7B',
    large_p3: 'Phi-3-Mini-3.8B',
    xl: 'Qwen2.5-7B',
    xl_mis: 'Zephyr-7B'
  };

  function getMethodName(m) {
    return METHOD_NAMES[m] || m.toUpperCase();
  }

  function getBackboneName(b) {
    return BACKBONE_NAMES[b] || b.replace('_', '-');
  }

  /* --------------------------------------------------------------------------
     2. Value Count-up Animation Helper
     ----------------------------------------------------------------------- */
  const currentMetricValues = {};

  function animateMetric(elementId, targetVal, decimals, unit, duration) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (prefersReducedMotion || duration === 0) {
      el.innerHTML = `${targetVal.toFixed(decimals)}${unit ? `<span class="metric-unit">${unit}</span>` : ''}`;
      currentMetricValues[elementId] = targetVal;
      return;
    }

    const startVal = currentMetricValues[elementId] !== undefined ? currentMetricValues[elementId] : targetVal;
    currentMetricValues[elementId] = targetVal;

    const startTime = performance.now();
    const animDur = duration || 220;

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(1, elapsed / animDur);
      // Easing: cubic-bezier(0.2, 0, 1, 1)
      const current = startVal + (targetVal - startVal) * progress;

      el.innerHTML = `${current.toFixed(decimals)}${unit ? `<span class="metric-unit">${unit}</span>` : ''}`;

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.innerHTML = `${targetVal.toFixed(decimals)}${unit ? `<span class="metric-unit">${unit}</span>` : ''}`;
      }
    }

    requestAnimationFrame(step);
  }

  /* --------------------------------------------------------------------------
     3. Recommendation Lab Update Logic
     ----------------------------------------------------------------------- */
  let heroChartInstance = null;

  function runRecommendation() {
    const query = {
      maxVramGb: state.vram,
      minAccuracy: state.accuracy,
      maxCarbon: state.carbon,
      maxTimeSeconds: state.runtimeMinutes * 60,
      profile: state.profile
    };

    if (state.backbone) query.backbones = [state.backbone];
    if (state.method) query.methods = [state.method];

    const result = Engine.recommend(query);

    const normalPanel = document.getElementById('recommendation-content');
    const infeasiblePanel = document.getElementById('infeasible-panel');
    const dropReasonsEl = document.getElementById('infeasible-reasons-list');

    if (!result.feasible || result.feasible.length === 0) {
      if (normalPanel) normalPanel.style.display = 'none';
      if (infeasiblePanel) {
        infeasiblePanel.classList.add('is-active');
        if (dropReasonsEl) {
          dropReasonsEl.innerHTML = '';
          (result.reasons || []).forEach(r => {
            const li = document.createElement('div');
            li.textContent = `· ${r}`;
            dropReasonsEl.appendChild(li);
          });
        }
      }
      return;
    }

    // Feasible candidates found
    if (infeasiblePanel) infeasiblePanel.classList.remove('is-active');
    if (normalPanel) normalPanel.style.display = 'block';

    const top = result.ranked[0];

    // Header & Titles
    const methodDisplay = getMethodName(top.method);
    const backboneDisplay = getBackboneName(top.backbone);
    const titleEl = document.getElementById('rec-title');
    if (titleEl) {
      titleEl.textContent = `${methodDisplay} · ${backboneDisplay}`;
    }

    // Formulate a research rationale
    const rationaleEl = document.getElementById('rec-rationale');
    if (rationaleEl) {
      let rationaleText = '';
      if (state.profile === 'strict_carbon') {
        rationaleText = `Lowest carbon emission candidate (${top.pred_carbon_kgco2eq.toFixed(3)} kgCO₂e) delivering ${(top.pred_accuracy * 100).toFixed(1)}% accuracy under the ${state.vram} GB memory ceiling.`;
      } else if (state.profile === 'high_accuracy') {
        rationaleText = `Maximizes downstream predictive performance (${(top.pred_accuracy * 100).toFixed(1)}% accuracy) while safely fitting within the ${state.vram} GB VRAM budget.`;
      } else {
        rationaleText = `Optimal multi-objective Pareto trade-off: ${(top.pred_accuracy * 100).toFixed(1)}% accuracy, ${top.pred_peak_vram_gb.toFixed(1)} GB peak memory, and minimal carbon intensity.`;
      }
      rationaleEl.textContent = rationaleText;
    }

    // Metrics count-up
    animateMetric('m-acc', top.pred_accuracy, 3, '');
    animateMetric('m-vram', top.pred_peak_vram_gb, 1, 'GB');
    animateMetric('m-energy', top.pred_energy_kwh, 4, 'kWh');
    animateMetric('m-carbon', top.pred_carbon_kgco2eq, 4, 'kg');
    animateMetric('m-time', top.pred_wall_clock_s / 60, 1, 'min');
    animateMetric('m-gei', top.gei, 3, '');

    const feasEl = document.getElementById('m-feas');
    if (feasEl) feasEl.textContent = '✓ FEASIBLE';

    const confEl = document.getElementById('m-conf');
    if (confEl) {
      confEl.textContent = top.confidence === 'HIGH' ? 'High' : (top.confidence === 'LOW' ? 'Low*' : 'Medium');
    }

    // Comparison Mini-Chart (Top 3 Candidates)
    const barsContainer = document.getElementById('comparison-bars');
    if (barsContainer) {
      barsContainer.innerHTML = '';
      const top3 = result.ranked.slice(0, 3);
      const topGei = top3[0].gei || 1;

      top3.forEach((cand, idx) => {
        const row = document.createElement('div');
        row.className = 'bar-row';
        row.setAttribute('tabindex', '0');
        row.setAttribute('role', 'button');
        row.setAttribute('aria-label', `${getMethodName(cand.method)} on ${getBackboneName(cand.backbone)}, GEI ${cand.gei.toFixed(3)}`);

        const label = document.createElement('div');
        label.className = 'bar-label';
        label.textContent = `${getMethodName(cand.method)} (${cand.backbone.replace('_', '-')})`;

        const track = document.createElement('div');
        track.className = 'bar-track';

        const fill = document.createElement('div');
        fill.className = `bar-fill ${idx === 0 ? 'is-selected' : ''}`;
        const pct = Math.max(12, Math.min(100, (cand.gei / topGei) * 100));
        fill.style.width = `${pct}%`;
        track.appendChild(fill);

        const val = document.createElement('div');
        val.className = 'bar-val';
        val.textContent = cand.gei.toFixed(3);

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(val);

        const candInfo = {
          name: `${getMethodName(cand.method)} · ${getBackboneName(cand.backbone)}`,
          acc: cand.pred_accuracy,
          vram: cand.pred_peak_vram_gb,
          gei: cand.gei
        };

        row.addEventListener('mouseenter', (e) => Charts.showTooltip(e, candInfo));
        row.addEventListener('focus', (e) => Charts.showTooltip(e, candInfo));
        row.addEventListener('mouseleave', Charts.hideTooltip);
        row.addEventListener('blur', Charts.hideTooltip);

        barsContainer.appendChild(row);
      });
    }

    // Alternatives List (2 rows with "why not")
    const altContainer = document.getElementById('alternatives-list');
    if (altContainer) {
      altContainer.innerHTML = '';
      const alts = result.ranked.slice(1, 3);

      if (alts.length === 0) {
        altContainer.innerHTML = '<div class="alt-row"><span class="alt-why">No alternative candidates pass all selected constraints.</span></div>';
      } else {
        alts.forEach(alt => {
          const altRow = document.createElement('div');
          altRow.className = 'alt-row';

          const nameSpan = document.createElement('span');
          nameSpan.className = 'alt-name';
          nameSpan.textContent = `${getMethodName(alt.method)} + ${getBackboneName(alt.backbone)}`;

          // Generate "why not" comparison relative to top candidate
          const whySpan = document.createElement('span');
          whySpan.className = 'alt-why';

          const accDiff = ((alt.pred_accuracy - top.pred_accuracy) * 100).toFixed(1);
          const vramDiff = (alt.pred_peak_vram_gb - top.pred_peak_vram_gb).toFixed(1);

          let whyText = '';
          if (alt.pred_accuracy < top.pred_accuracy) {
            whyText += `${Math.abs(accDiff)}% lower accuracy`;
          } else {
            whyText += `higher memory consumption (+${vramDiff} GB)`;
          }

          if (alt.pred_carbon_kgco2eq > top.pred_carbon_kgco2eq) {
            whyText += `, higher carbon intensity`;
          } else if (alt.pred_peak_vram_gb < top.pred_peak_vram_gb) {
            whyText += `, saves ${Math.abs(vramDiff)} GB VRAM`;
          }

          whySpan.textContent = `— ${whyText}`;

          const geiSpan = document.createElement('span');
          geiSpan.className = 'alt-gei';
          geiSpan.textContent = `GEI ${alt.gei.toFixed(3)}`;

          altRow.appendChild(nameSpan);
          altRow.appendChild(whySpan);
          altRow.appendChild(geiSpan);
          altContainer.appendChild(altRow);
        });
      }
    }
  }

  /* --------------------------------------------------------------------------
     4. Setup Lab Controls & Event Listeners
     ----------------------------------------------------------------------- */
  function setupLabControls() {
    // VRAM Slider
    const vramSlider = document.getElementById('sl-vram');
    const vramVal = document.getElementById('sl-vram-val');
    if (vramSlider) {
      vramSlider.addEventListener('input', (e) => {
        state.vram = parseFloat(e.target.value);
        if (vramVal) vramVal.textContent = `${state.vram} GB`;
        if (heroChartInstance && heroChartInstance.setVram) {
          heroChartInstance.setVram(state.vram);
        }
        runRecommendation();
      });
    }

    // Accuracy Slider
    const accSlider = document.getElementById('sl-acc');
    const accVal = document.getElementById('sl-acc-val');
    if (accSlider) {
      accSlider.addEventListener('input', (e) => {
        state.accuracy = parseFloat(e.target.value);
        if (accVal) accVal.textContent = state.accuracy.toFixed(2);
        runRecommendation();
      });
    }

    // Carbon Slider
    const carbonSlider = document.getElementById('sl-carbon');
    const carbonVal = document.getElementById('sl-carbon-val');
    if (carbonSlider) {
      carbonSlider.addEventListener('input', (e) => {
        state.carbon = parseFloat(e.target.value);
        if (carbonVal) carbonVal.textContent = `${state.carbon.toFixed(2)} kg`;
        runRecommendation();
      });
    }

    // Runtime Slider
    const timeSlider = document.getElementById('sl-time');
    const timeVal = document.getElementById('sl-time-val');
    if (timeSlider) {
      timeSlider.addEventListener('input', (e) => {
        state.runtimeMinutes = parseInt(e.target.value, 10);
        if (timeVal) timeVal.textContent = `${state.runtimeMinutes} min`;
        runRecommendation();
      });
    }

    // Profile Segmented Control
    const profileBtns = document.querySelectorAll('.profile-btn');
    profileBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        profileBtns.forEach(b => b.setAttribute('aria-pressed', 'false'));
        btn.setAttribute('aria-pressed', 'true');
        state.profile = btn.getAttribute('data-profile');
        runRecommendation();
      });
    });

    // Dropdowns
    const bbSelect = document.getElementById('sel-backbone');
    if (bbSelect) {
      bbSelect.addEventListener('change', (e) => {
        state.backbone = e.target.value;
        runRecommendation();
      });
    }

    const methodSelect = document.getElementById('sel-method');
    if (methodSelect) {
      methodSelect.addEventListener('change', (e) => {
        state.method = e.target.value;
        runRecommendation();
      });
    }

    // Relax Constraints Button (Infeasible Panel)
    const relaxBtn = document.getElementById('relax-constraints-btn');
    if (relaxBtn) {
      relaxBtn.addEventListener('click', () => {
        state.vram = 24;
        state.accuracy = 0.85;
        state.carbon = 0.50;
        state.runtimeMinutes = 120;
        state.backbone = '';
        state.method = '';

        if (vramSlider) { vramSlider.value = 24; vramVal.textContent = '24 GB'; }
        if (accSlider) { accSlider.value = 0.85; accVal.textContent = '0.85'; }
        if (carbonSlider) { carbonSlider.value = 0.50; carbonVal.textContent = '0.50 kg'; }
        if (timeSlider) { timeSlider.value = 120; timeVal.textContent = '120 min'; }
        if (bbSelect) bbSelect.value = '';
        if (methodSelect) methodSelect.value = '';

        if (heroChartInstance) heroChartInstance.setVram(24);
        runRecommendation();
      });
    }
  }

  /* --------------------------------------------------------------------------
     5. Section 3: Why This Matters Tabs & Interactive Visuals
     ----------------------------------------------------------------------- */
  function setupWhySection() {
    const tabs = document.querySelectorAll('.why-tab-btn');
    const cards = document.querySelectorAll('.why-step-card');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const step = tab.getAttribute('data-step');
        tabs.forEach(t => t.setAttribute('aria-selected', 'false'));
        tab.setAttribute('aria-selected', 'true');

        cards.forEach(c => {
          if (c.getAttribute('data-step') === step) {
            c.classList.add('is-active');
          } else {
            c.classList.remove('is-active');
          }
        });
      });
    });

    // Beat 3: Two-state toggle (Run all vs GreenPEFT)
    const filterContainer = document.getElementById('filter-dots-mount');
    if (filterContainer) {
      const filterSvgInstance = Charts.initFilterToggleSvg(filterContainer);

      const toggleRunAll = document.getElementById('btn-run-all');
      const toggleGreenPeft = document.getElementById('btn-filter-greenpeft');

      if (toggleRunAll && toggleGreenPeft) {
        toggleRunAll.addEventListener('click', () => {
          toggleRunAll.setAttribute('aria-pressed', 'true');
          toggleGreenPeft.setAttribute('aria-pressed', 'false');
          filterSvgInstance.setState('run_all');
        });

        toggleGreenPeft.addEventListener('click', () => {
          toggleGreenPeft.setAttribute('aria-pressed', 'true');
          toggleRunAll.setAttribute('aria-pressed', 'false');
          filterSvgInstance.setState('greenpeft');
        });
      }
    }
  }

  /* --------------------------------------------------------------------------
     6. Section 4: Pipeline Diagram (Expandable Stages & I/O Chips)
     ----------------------------------------------------------------------- */
  const PIPELINE_DATA = {
    harmonize: {
      inputs: ['raw_telemetry', 'hardware_logs', 'seed_variance'],
      outputs: ['unified_metrics', 'calibration_envelope'],
      desc: 'Normalizes heterogeneous training traces and hardware telemetry into standard units (FLOPs, peak memory GB, kWh, wall-clock seconds). Eliminates measurement divergence across different tracking libraries.'
    },
    benchmark: {
      inputs: ['backbone_params', 'adapter_rank', 'quant_bits', 'dataset_size'],
      outputs: ['60_run_telemetry', 'empirical_bounds'],
      desc: 'Executes calibrated micro-benchmarks across a 60-run parameter envelope on NVIDIA Tesla T4 GPUs. Captures empirical multi-objective ground truth across 0.5B to 3B parameter architectures.'
    },
    surrogate: {
      inputs: ['config_features', 'model_family', 'target_hardware'],
      outputs: ['accuracy', 'peak_vram', 'energy_kwh', 'wall_clock_s'],
      desc: 'Zero-shot regression models infer accuracy, peak VRAM, energy consumption, and wall-clock runtime instantly from configuration metadata. Eliminates blind parameter sweeps before code execution.'
    },
    filter: {
      inputs: ['vram_budget', 'accuracy_floor', 'carbon_limit', 'time_ceiling'],
      outputs: ['feasible_set', 'non_dominated_pareto_front'],
      desc: 'Filters candidate configurations against hardware ceilings and user accuracy constraints. Extracts the non-dominated Pareto frontier, discarding all suboptimal trade-offs.'
    },
    recommend: {
      inputs: ['preference_weights', 'gei_profile'],
      outputs: ['gei_ranking', 'top_shortlist', 'inspectable_rationale'],
      desc: 'Computes the Green Efficiency Index (GEI) across customizable preference profiles to deliver an inspectable shortlist and actionable candidate recommendation.'
    }
  };

  function setupPipeline() {
    const stages = document.querySelectorAll('.pipeline-stage');
    const chipsMount = document.getElementById('pipeline-chips-mount');
    const descMount = document.getElementById('pipeline-desc-mount');

    function updateStageInfo(stageKey) {
      const data = PIPELINE_DATA[stageKey] || PIPELINE_DATA.harmonize;

      if (chipsMount) {
        let html = '';
        data.inputs.forEach(inp => {
          html += `<span class="chip-tag">${inp}</span>`;
        });
        html += `<span class="chip-arrow">→</span>`;
        data.outputs.forEach(outp => {
          html += `<span class="chip-tag" style="background:#000;color:#fff;">${outp}</span>`;
        });
        chipsMount.innerHTML = html;
      }

      if (descMount) {
        descMount.textContent = data.desc;
      }
    }

    stages.forEach(stage => {
      const key = stage.getAttribute('data-stage');

      stage.addEventListener('mouseenter', () => updateStageInfo(key));
      stage.addEventListener('focus', () => updateStageInfo(key));

      stage.addEventListener('click', () => {
        stages.forEach(s => s.classList.remove('is-active'));
        stage.classList.add('is-active');
        updateStageInfo(key);
      });
    });

    // Default stage
    updateStageInfo('harmonize');
  }

  /* --------------------------------------------------------------------------
     7. Section 5: Accordion (Accessible Keyboard Navigation)
     ----------------------------------------------------------------------- */
  function setupAccordion() {
    const triggers = document.querySelectorAll('.accordion-trigger');

    triggers.forEach(trigger => {
      trigger.addEventListener('click', () => {
        const isExpanded = trigger.getAttribute('aria-expanded') === 'true';
        trigger.setAttribute('aria-expanded', String(!isExpanded));
        const content = trigger.nextElementSibling;
        if (content) {
          content.style.display = isExpanded ? 'none' : 'block';
        }
      });
    });
  }

  /* --------------------------------------------------------------------------
     8. Section 6: CLI Quickstart (Copy Button & Typing Animation)
     ----------------------------------------------------------------------- */
  function setupCliTerminal() {
    const copyBtn = document.getElementById('cli-copy-btn');
    const cliCommandText = `$ pip install green-peft\n$ green-peft recommend --artifacts-dir ./peft_bench_export --vram 16 --accuracy 0.90 --profile balanced`;

    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        if (navigator.clipboard) {
          navigator.clipboard.writeText(cliCommandText).then(() => {
            copyBtn.textContent = 'Copied ✓';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 2000);
          });
        }
      });
    }

    const runDemoBtn = document.getElementById('cli-run-demo-btn');
    const terminalOutput = document.getElementById('terminal-output');

    const terminalLines = [
      '→ Filtering 47 candidates...',
      '→ 6 feasible under constraints.',
      '→ Pareto frontier: 3 candidates.',
      '',
      'RECOMMENDED',
      '  method      LISA',
      '  backbone    Qwen2.5-Medium (1.5B)',
      '  accuracy    0.923 ± 0.011',
      '  peak_vram   14.2 GB',
      '  energy      0.18 kWh',
      '  carbon      0.09 kgCO₂e',
      '  runtime     42 min',
      '  GEI         0.847',
      '',
      'Alternatives:',
      '  QLoRA + TinyLlama-Small   GEI 0.791',
      '  LoRA-FA + Qwen2.5-Tiny    GEI 0.744'
    ];

    function playTypingAnimation() {
      if (!terminalOutput) return;

      if (prefersReducedMotion) {
        terminalOutput.textContent = terminalLines.join('\n');
        return;
      }

      terminalOutput.textContent = '';
      let lineIndex = 0;

      function typeLine() {
        if (lineIndex < terminalLines.length) {
          terminalOutput.textContent += (lineIndex > 0 ? '\n' : '') + terminalLines[lineIndex];
          lineIndex++;
          setTimeout(typeLine, 80);
        }
      }

      typeLine();
    }

    if (runDemoBtn) {
      runDemoBtn.addEventListener('click', playTypingAnimation);
    }
  }

  /* --------------------------------------------------------------------------
     9. Section 7: BibTeX Copy Button
     ----------------------------------------------------------------------- */
  function setupBibtex() {
    const bibtexCopyBtn = document.getElementById('bibtex-copy-btn');
    const bibtexCodeEl = document.getElementById('bibtex-text');

    if (bibtexCopyBtn && bibtexCodeEl) {
      bibtexCopyBtn.addEventListener('click', () => {
        if (navigator.clipboard) {
          navigator.clipboard.writeText(bibtexCodeEl.textContent).then(() => {
            bibtexCopyBtn.textContent = 'Copied ✓';
            setTimeout(() => { bibtexCopyBtn.textContent = 'Copy BibTeX'; }, 2000);
          });
        }
      });
    }
  }

  /* --------------------------------------------------------------------------
     10. Initialization
     ----------------------------------------------------------------------- */
  window.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Hero Scatter
    const heroMount = document.getElementById('hero-scatter-mount');
    if (heroMount) {
      heroChartInstance = Charts.initHeroScatter(heroMount, (newVram) => {
        state.vram = Math.round(newVram);
        const vramSlider = document.getElementById('sl-vram');
        const vramVal = document.getElementById('sl-vram-val');
        if (vramSlider) vramSlider.value = state.vram;
        if (vramVal) vramVal.textContent = `${state.vram} GB`;
        runRecommendation();
      });
    }

    // 2. Initialize Evidence Scatter
    const evidenceMount = document.getElementById('evidence-scatter-mount');
    if (evidenceMount) {
      Charts.initEvidenceScatter(evidenceMount);
    }

    // 3. Setup Lab & Sections
    setupLabControls();
    setupWhySection();
    setupPipeline();
    setupAccordion();
    setupCliTerminal();
    setupBibtex();

    // 4. Initial Recommender Run
    runRecommendation();
  });

})();
