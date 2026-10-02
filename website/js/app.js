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

  function updateMascot(isFeasible, isFocused) {
    const mascot = document.getElementById('greenpeft-mascot');
    const status = document.getElementById('greenpeft-mascot-status');
    if (!mascot) return;

    mascot.setAttribute('data-state', isFeasible ? 'feasible' : 'infeasible');
    mascot.classList.toggle('is-focused', Boolean(isFocused));
    mascot.setAttribute('aria-label', isFeasible
      ? 'GreenPEFT Bot: recommendation ready'
      : 'GreenPEFT Bot: no feasible candidate');
    if (status) status.textContent = isFeasible ? 'Ready' : 'Needs adjustment';
  }

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
    const outputPanel = document.getElementById('rec-output-panel');

    const normalPanel = document.getElementById('recommendation-content');
    const infeasiblePanel = document.getElementById('infeasible-panel');
    const dropReasonsEl = document.getElementById('infeasible-reasons-list');

    if (!result.feasible || result.feasible.length === 0) {
      updateMascot(false, false);
      if (outputPanel) outputPanel.setAttribute('data-result-status', 'infeasible');
      if (normalPanel) normalPanel.style.display = 'none';
      const barsContainer = document.getElementById('comparison-bars');
      const altContainer = document.getElementById('alternatives-list');
      if (barsContainer) barsContainer.replaceChildren();
      if (altContainer) altContainer.innerHTML = '<div class="alt-row"><span class="alt-why">No alternative candidates pass all selected constraints.</span></div>';
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
    updateMascot(true, false);
    if (outputPanel) outputPanel.setAttribute('data-result-status', 'ready');
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
        row.setAttribute('aria-current', idx === 0 ? 'true' : 'false');
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
        row.addEventListener('click', () => {
          barsContainer.querySelectorAll('.bar-row').forEach(item => item.setAttribute('aria-current', 'false'));
          row.setAttribute('aria-current', 'true');
          if (outputPanel) outputPanel.setAttribute('data-selected-candidate', candInfo.name);
        });
        row.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            row.click();
          }
        });

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
    function markMascotFocused() {
      updateMascot(true, true);
      window.clearTimeout(markMascotFocused.timer);
      markMascotFocused.timer = window.setTimeout(() => {
        const panel = document.getElementById('rec-output-panel');
        updateMascot(panel && panel.getAttribute('data-result-status') === 'ready', false);
      }, 500);
    }

    function trackMascotEyes(event) {
      const mascot = document.getElementById('greenpeft-mascot');
      if (!mascot) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width - 0.5) * 4;
      const y = ((event.clientY - rect.top) / rect.height - 0.5) * 3;
      mascot.style.setProperty('--mascot-eye-x', `${x.toFixed(2)}px`);
      mascot.style.setProperty('--mascot-eye-y', `${y.toFixed(2)}px`);
    }

    function resetMascotEyes() {
      const mascot = document.getElementById('greenpeft-mascot');
      if (mascot) {
        mascot.style.setProperty('--mascot-eye-x', '0px');
        mascot.style.setProperty('--mascot-eye-y', '0px');
      }
    }

    document.querySelectorAll('.lab-controls-sticky input, .lab-controls-sticky select').forEach(control => {
      control.addEventListener('pointermove', trackMascotEyes);
      control.addEventListener('pointerleave', resetMascotEyes);
    });

    // VRAM Slider
    const vramSlider = document.getElementById('sl-vram');
    const vramVal = document.getElementById('sl-vram-val');
    if (vramSlider) {
      vramSlider.addEventListener('input', (e) => {
        markMascotFocused();
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
        markMascotFocused();
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
        markMascotFocused();
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
        markMascotFocused();
        state.runtimeMinutes = parseInt(e.target.value, 10);
        if (timeVal) timeVal.textContent = `${state.runtimeMinutes} min`;
        runRecommendation();
      });
    }

    // Profile Segmented Control
    const profileBtns = document.querySelectorAll('.profile-btn');
    profileBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        markMascotFocused();
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
        markMascotFocused();
        state.backbone = e.target.value;
        runRecommendation();
      });
    }

    const methodSelect = document.getElementById('sel-method');
    if (methodSelect) {
      methodSelect.addEventListener('change', (e) => {
        markMascotFocused();
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
    benchmark: {
      inputs: ['4 backbones', '5 PEFT methods', 'SST-2', 'Tesla T4 / Kaggle', '300 steps × 3 seeds'],
      outputs: ['Accuracy (%)', 'Peak VRAM (GB)', 'Energy (W) @ NVML 5Hz', 'Runtime (s)', 'CO₂e = Energy × 0.65 kg/kWh'],
      desc: 'Benchmark: Qwen2.5-0.5B, TinyLlama-1.1B, Qwen2.5-1.5B, and Qwen2.5-3B are evaluated with Full FT, LoRA, QLoRA, LoRA-FA, and LISA on SST-2 using a fixed 300-step, 3-seed setup. We collect Accuracy (%), Peak VRAM (GB), Energy (W) via NVML at 5 Hz, Runtime (s), and operational CO₂e derived as Energy × 0.65 kg/kWh.'
    },
    audit: {
      inputs: ['Pass 1: ~59.9 W under load', 'Pass 2: ~9.97 W idle floor excluded', '41 valid configurations'],
      outputs: ['14 / 20 matrix cells covered', 'canonical dataset', 'coverage matrix'],
      desc: 'Audit & Coverage: Pass 1 retained the approximately 59.9 W under-load telemetry. Pass 2 was excluded because the approximately 9.97 W idle-floor telemetry exposed an instrumentation bug. The canonical dataset contains 41 valid configurations, covering 14 of 20 backbone–method matrix cells.',
      matrix: true
    },
    modeling: {
      inputs: ['22 pre-run metadata features', 'Scale', 'Bit width', 'Adapter rank', 'Memory / weight bytes'],
      outputs: ['Ridge: Accuracy', 'Ridge: Memory / VRAM', 'Ridge: Energy', 'Ridge: Runtime'],
      desc: 'Modeling: four Ridge surrogate models consume 22 pre-run metadata features covering scale, bit width, adapter rank, memory bytes, and weight bytes. Validation uses a hierarchical pyramid: pass-grouped, seed-grouped, and leave-one-tier-out (LOTO). Status gate: Energy/Carbon OK (R² = 0.90, MAPE = 10.6%); Memory, Runtime, and Accuracy remain preliminary.'
    },
    decision: {
      inputs: ['70 candidates', 'sanity filter', 'VRAM + 10% safety margin', 'accuracy / carbon / time limits'],
      outputs: ['4-objective Pareto frontier', 'GEI ranking', 'Measured (High) / Interpolated (Low)', 'DOI · HF · Kaggle · CLI'],
      desc: 'Decision Engine: 70 candidates pass through a sanity filter, constraint filter (Max VRAM with a 10% safety margin, Accuracy Floor, Carbon and Time limits), four-objective Pareto frontier, and Green Efficiency Index (GEI) ranking. Outputs label evidence as Measured (High) or Interpolated / Out-of-bounds (Low), alongside open-science artifacts: DOIs, the Hugging Face model bundle, Kaggle dataset, and pip install green-peft.'
    },
  };

  function setupPipeline() {
    const stages = document.querySelectorAll('.pipeline-stage');
    const chipsMount = document.getElementById('pipeline-chips-mount');
    const descMount = document.getElementById('pipeline-desc-mount');

    function updateStageInfo(stageKey) {
      const data = PIPELINE_DATA[stageKey] || PIPELINE_DATA.benchmark;

      if (chipsMount) {
        let html = '';
        data.inputs.forEach(inp => {
          html += `<span class="chip-tag">${inp}</span>`;
        });
        html += `<span class="chip-arrow">→</span>`;
        data.outputs.forEach(outp => {
          html += `<span class="chip-tag chip-tag--output">${outp}</span>`;
        });
        chipsMount.innerHTML = html;
        if (data.matrix) {
          html += '<div class="coverage-matrix" aria-label="Coverage matrix showing 14 of 20 matrix cells covered">' +
            '<span class="coverage-matrix-title">Coverage matrix · 14 / 20 cells</span>' +
            '<span class="coverage-row"><b>Backbone</b><b>Full FT</b><b>LoRA</b><b>QLoRA</b><b>LoRA-FA</b><b>LISA</b></span>' +
            '<span class="coverage-row"><b>0.5B</b><i>●</i><i>●</i><i>●</i><i>●</i><i>●</i></span>' +
            '<span class="coverage-row"><b>1.1B</b><i>●</i><i>●</i><i>●</i><i>●</i><i>●</i></span>' +
            '<span class="coverage-row"><b>1.5B</b><i>○</i><i>●</i><i>●</i><i>○</i><i>●</i></span>' +
            '<span class="coverage-row"><b>3.0B</b><i>○</i><i>●</i><i>●</i><i>○</i><i>○</i></span>' +
            '</div>';
          chipsMount.innerHTML = html;
        }
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
        stages.forEach(s => s.setAttribute('aria-pressed', 'false'));
        stage.classList.add('is-active');
        stage.setAttribute('aria-pressed', 'true');
        updateStageInfo(key);
      });

      stage.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          stage.click();
        }
      });
    });

    // Default stage
    updateStageInfo('benchmark');
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

    const terminalLines = String.raw`   ____                     ____  _____ _____ _____
  / ___|_ __ ___  ___ _ __ |  _ \| ____|  ___|_   _|
 | |  _| '__/ _ \/ _ \ '_ \| |_) |  _| | |_    | |
 | |_| | | |  __/  __/ | | |  __/| |___|  _|   | |
  \____|_|  \___|\___|_| |_|_|   |_____|_|     |_|

  GreenPEFT v0.3.0 - sustainability-aware PEFT decision support
  Ashraful Islam Tanzil | United International University
  https://github.com/ai-tanzil811
  surrogate status: [ok] VALIDATED
  data doi 10.34740/KAGGLE/DSV/20178095 | model doi 10.57967/hf/10690

→ Filtering 47 candidates...
→ 6 feasible under constraints.
→ Pareto frontier: 3 candidates.

RECOMMENDED
  method      LISA
  backbone    Qwen2.5-Medium (1.5B)
  accuracy    0.923 ± 0.011
  peak_vram   14.2 GB
  energy      0.18 kWh
  carbon      0.09 kgCO₂e
  runtime     42 min
  GEI         0.847

Alternatives:
  QLoRA + TinyLlama-Small   GEI 0.791
  LoRA-FA + Qwen2.5-Tiny    GEI 0.744`.split('\n');
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
    const hubBibtexCopyBtn = document.getElementById('bibtex-copy-btn-hub');
    const installCopyBtn = document.getElementById('install-copy-btn');
    const bibtexCodeEl = document.getElementById('bibtex-text');

    function copyText(text, button, defaultLabel) {
      if (!navigator.clipboard || !text || !button) return;
      navigator.clipboard.writeText(text).then(() => {
        button.textContent = 'Copied ✓';
        button.setAttribute('aria-label', 'Copied to clipboard');
        setTimeout(() => {
          button.textContent = defaultLabel;
          button.removeAttribute('aria-label');
        }, 2000);
      }).catch(() => {
        button.textContent = 'Copy unavailable';
        setTimeout(() => { button.textContent = defaultLabel; }, 2000);
      });
    }

    if (bibtexCodeEl) {
      [bibtexCopyBtn, hubBibtexCopyBtn].filter(Boolean).forEach(button => {
        button.addEventListener('click', () => copyText(bibtexCodeEl.textContent, button, 'Copy BibTeX'));
      });
    }
    if (installCopyBtn) {
      installCopyBtn.addEventListener('click', () => copyText(installCopyBtn.dataset.copyText, installCopyBtn, 'Copy install command'));
    }

    document.querySelectorAll('[data-resource-tab]').forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.resourceTab;
        document.querySelectorAll('[data-resource-tab]').forEach(item => {
          const active = item === tab;
          item.classList.toggle('is-active', active);
          item.setAttribute('aria-selected', String(active));
        });
        document.querySelectorAll('.resource-panel').forEach(panel => {
          panel.hidden = panel.id !== `resources-${target}`;
          panel.classList.toggle('is-active', panel.id === `resources-${target}`);
        });
      });
    });
  }

  /* --------------------------------------------------------------------------
     10. Theme Toggle System (Default Dark Mode with Instant Chart Sync)
     ----------------------------------------------------------------------- */
  function setupThemeToggle() {
    const themeButtons = document.querySelectorAll('.theme-switch-btn');

    let currentTheme = localStorage.getItem('greenpeft_theme') || 'dark';
    applyTheme(currentTheme);

    function applyTheme(theme) {
      currentTheme = theme === 'light' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', theme);
      localStorage.setItem('greenpeft_theme', theme);

      themeButtons.forEach(button => {
        const isActive = button.getAttribute('data-theme-val') === currentTheme;
        button.classList.toggle('is-active', isActive);
        button.setAttribute('aria-pressed', String(isActive));
      });

      if (Charts && Charts.refreshTheme) {
        Charts.refreshTheme();
      }
    }

    themeButtons.forEach(button => {
      button.addEventListener('click', () => {
        applyTheme(button.getAttribute('data-theme-val'));
      });
    });
  }

  /* --------------------------------------------------------------------------
     11. Language Selector (English, Bengali, Chinese, Russian)
     ----------------------------------------------------------------------- */
  const LANGUAGE_STRINGS = {
    en: {
      navRecommender: '01 Recommender', navWhy: '02 Why', navArchitecture: '03 Architecture',
      navEvidence: '04 Evidence', navTrust: '05 Trust', navCli: '06 CLI', navResearch: '07 Research',
      github: 'GitHub', dark: 'Dark', light: 'Light', language: 'Language',
      tryRecommender: 'Try the recommender', exploreResearch: 'Explore the research',
      maxVram: 'Max VRAM', minAccuracy: 'Min Accuracy', carbonBudget: 'Carbon Budget',
      runtimeLimit: 'Runtime Limit', hardwareObjectives: 'Hardware & Objectives',
      decisionEngine: '01 / Decision Engine', why: '02 / Motivation', architecture: '03 / Architecture',
      trust: '04 / Transparency', copy: 'Copy BibTeX', runDemo: 'Run demo'
    },
    bn: {
      navRecommender: '০১ সুপারিশ', navWhy: '০২ কেন', navArchitecture: '০৩ স্থাপত্য',
      navEvidence: '০৪ প্রমাণ', navTrust: '০৫ বিশ্বাস', navCli: '০৬ CLI', navResearch: '০৭ গবেষণা',
      github: 'গিটহাব', dark: 'ডার্ক', light: 'লাইট', language: 'ভাষা',
      tryRecommender: 'সুপারিশ চালান', exploreResearch: 'গবেষণা দেখুন',
      maxVram: 'সর্বোচ্চ VRAM', minAccuracy: 'ন্যূনতম নির্ভুলতা', carbonBudget: 'কার্বন বাজেট',
      runtimeLimit: 'রানটাইম সীমা', hardwareObjectives: 'হার্ডওয়্যার ও লক্ষ্য',
      decisionEngine: '০১ / সিদ্ধান্ত ইঞ্জিন', why: '০২ / কেন', architecture: '০৩ / স্থাপত্য',
      trust: '০৪ / স্বচ্ছতা', copy: 'BibTeX কপি', runDemo: 'ডেমো চালান'
    },
    zh: {
      navRecommender: '01 推荐器', navWhy: '02 为什么', navArchitecture: '03 架构',
      navEvidence: '04 证据', navTrust: '05 可信度', navCli: '06 CLI', navResearch: '07 研究',
      github: 'GitHub', dark: '深色', light: '浅色', language: '语言',
      tryRecommender: '试用推荐器', exploreResearch: '探索研究',
      maxVram: '最大显存', minAccuracy: '最低准确率', carbonBudget: '碳预算',
      runtimeLimit: '运行时间限制', hardwareObjectives: '硬件与目标',
      decisionEngine: '01 / 决策引擎', why: '02 / 动机', architecture: '03 / 架构',
      trust: '04 / 透明度', copy: '复制 BibTeX', runDemo: '运行演示'
    },
    ru: {
      navRecommender: '01 Рекомендатор', navWhy: '02 Зачем', navArchitecture: '03 Архитектура',
      navEvidence: '04 Доказательства', navTrust: '05 Доверие', navCli: '06 CLI', navResearch: '07 Исследование',
      github: 'GitHub', dark: 'Тёмная', light: 'Светлая', language: 'Язык',
      tryRecommender: 'Запустить рекомендатор', exploreResearch: 'Изучить исследование',
      maxVram: 'Макс. VRAM', minAccuracy: 'Мин. точность', carbonBudget: 'Углеродный бюджет',
      runtimeLimit: 'Ограничение времени', hardwareObjectives: 'Оборудование и цели',
      decisionEngine: '01 / Движок решений', why: '02 / Мотивация', architecture: '03 / Архитектура',
      trust: '04 / Прозрачность', copy: 'Копировать BibTeX', runDemo: 'Запустить демо'
    }
  };

  function setupLanguageSelector() {
    const selector = document.getElementById('language-select');
    if (!selector) return;

    const translations = {
      '.nav-link[href="#recommender"]': 'navRecommender',
      '.nav-link[href="#why"]': 'navWhy',
      '.nav-link[href="#pipeline"]': 'navArchitecture',
      '.nav-link[href="#evidence"]': 'navEvidence',
      '.nav-link[href="#trust"]': 'navTrust',
      '.nav-link[href="#cli"]': 'navCli',
      '.nav-link[href="#research"]': 'navResearch',
      '.nav-actions > .btn': 'github',
      '#btn-theme-dark': 'dark',
      '#btn-theme-light': 'light',
      '.hero-actions .btn--primary': 'tryRecommender',
      '.hero-actions .btn--ghost': 'exploreResearch',
      '.lab-controls-title': 'hardwareObjectives',
      'label[for="sl-vram"]': 'maxVram',
      'label[for="sl-acc"]': 'minAccuracy',
      'label[for="sl-carbon"]': 'carbonBudget',
      'label[for="sl-time"]': 'runtimeLimit',
      '#recommender .section-header .eyebrow': 'decisionEngine',
      '#why .section-header .eyebrow': 'why',
      '#pipeline .section-header .eyebrow': 'architecture',
      '#trust .section-header .eyebrow': 'trust',
      '#bibtex-copy-btn': 'copy',
      '#cli-run-demo-btn': 'runDemo',
      '#language-select': 'language'
    };

    function applyLanguage(language) {
      const strings = LANGUAGE_STRINGS[language] || LANGUAGE_STRINGS.en;
      Object.keys(translations).forEach(selectorText => {
        const element = document.querySelector(selectorText);
        const key = translations[selectorText];
        if (!element || !strings[key]) return;
        if (element.id === 'language-select') {
          element.setAttribute('aria-label', strings[key]);
        } else if (element.classList.contains('theme-switch-btn')) {
          const textNode = Array.from(element.childNodes).reverse().find(node => node.nodeType === Node.TEXT_NODE);
          if (textNode) textNode.textContent = ` ${strings[key]}`;
        } else {
          element.textContent = strings[key];
        }
      });
      document.documentElement.lang = language === 'bn' ? 'bn' : (language === 'zh' ? 'zh-CN' : language);
      selector.value = LANGUAGE_STRINGS[language] ? language : 'en';
      localStorage.setItem('greenpeft_language', selector.value);
    }

    applyLanguage(localStorage.getItem('greenpeft_language') || 'en');
    selector.addEventListener('change', () => applyLanguage(selector.value));
  }

  /* --------------------------------------------------------------------------
     11. Initialization
     ----------------------------------------------------------------------- */
  let appInitialized = false;

  function initializeApp() {
    if (appInitialized) return;
    appInitialized = true;
    // Setup Theme (Default Dark Mode)
    setupThemeToggle();
    setupLanguageSelector();

    // 1. Initialize Hero Scatter
    const heroMount = document.getElementById('hero-scatter-mount');
    if (heroMount) {
      heroChartInstance = Charts.initHeroScatter(heroMount, (newVram) => {
        updateMascot(true, true);
        window.clearTimeout(updateMascot.heroFocusTimer);
        updateMascot.heroFocusTimer = window.setTimeout(() => {
          const panel = document.getElementById('rec-output-panel');
          updateMascot(panel && panel.getAttribute('data-result-status') === 'ready', false);
        }, 500);
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
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initializeApp, { once: true });
  } else {
    initializeApp();
  }

})();
