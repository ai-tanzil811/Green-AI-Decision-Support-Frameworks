/* ===========================================================================
   GreenPEFT decision engine — browser port.

   This mirrors green_peft/recommender.py exactly: the plausibility drop, the
   VRAM feasibility gate with its safety margin, the carbon / accuracy / time
   constraints, the evidence floor, the non-dominated test and the min-max GEI
   normalisation. Only the surrogate inference happens elsewhere — those
   predictions arrive precomputed in data/candidates.js.

   Keep this file in step with recommender.py. If the Python changes, the page
   is wrong until this changes too.
   ========================================================================= */

window.GPEngine = (function () {
  'use strict';

  const FALLBACK_DATA = {
    model_status: 'FALLBACK — generated catalogue unavailable',
    engine: {
      vram_safety_margin: 0.1,
      gei_profiles: {
        balanced: [0.35, 0.25, 0.25, 0.15],
        strict_carbon: [0.2, 0.2, 0.5, 0.1],
        high_accuracy: [0.6, 0.15, 0.15, 0.1]
      }
    },
    method_labels: {
      full_ft: 'Full fine-tuning',
      lora: 'LoRA',
      qlora: 'QLoRA',
      lora_fa: 'LoRA-FA',
      lisa: 'LISA'
    },
    candidates: [
      { model_id: 'Qwen2.5-0.5B', backbone: 'tiny', params_b: 0.5, family: 'Qwen', method: 'lora', pred_accuracy: 0.925, pred_peak_vram_gb: 4.2, pred_carbon_kgco2eq: 0.012, pred_wall_clock_s: 480, confidence: 'HIGH', scope: 'MEASURED', implausible: false },
      { model_id: 'Qwen2.5-0.5B', backbone: 'tiny', params_b: 0.5, family: 'Qwen', method: 'qlora', pred_accuracy: 0.918, pred_peak_vram_gb: 3.4, pred_carbon_kgco2eq: 0.009, pred_wall_clock_s: 420, confidence: 'HIGH', scope: 'MEASURED', implausible: false },
      { model_id: 'TinyLlama-1.1B', backbone: 'small', params_b: 1.1, family: 'TinyLlama', method: 'lora', pred_accuracy: 0.94, pred_peak_vram_gb: 8.9, pred_carbon_kgco2eq: 0.019, pred_wall_clock_s: 650, confidence: 'HIGH', scope: 'MEASURED', implausible: false },
      { model_id: 'Qwen2.5-1.5B', backbone: 'medium', params_b: 1.5, family: 'Qwen', method: 'qlora', pred_accuracy: 0.952, pred_peak_vram_gb: 11.8, pred_carbon_kgco2eq: 0.027, pred_wall_clock_s: 820, confidence: 'HIGH', scope: 'MEASURED', implausible: false },
      { model_id: 'Qwen2.5-3B', backbone: 'large', params_b: 3, family: 'Qwen', method: 'lora', pred_accuracy: 0.961, pred_peak_vram_gb: 18.6, pred_carbon_kgco2eq: 0.044, pred_wall_clock_s: 1250, confidence: 'MEDIUM', scope: 'INTERPOLATED_SCALE', implausible: false }
    ]
  };
  const DATA = window.GREENPEFT_CANDIDATES || FALLBACK_DATA;
  const CFG = DATA.engine;

  const CONFIDENCE_ORDER = ['LOW', 'MEDIUM', 'HIGH'];

  // Weakest-to-strongest, same list the confidence module uses. Reversed when
  // tallying a feasible set so the strongest evidence is reported first.
  const SCOPE_ORDER = [
    'OUT_OF_RANGE_SCALE',
    'UNSEEN_FAMILY',
    'UNSEEN_METHOD_SCALE',
    'INTERPOLATED_SCALE',
    'MEASURED'
  ];

  const ALL = DATA.candidates.map(function (c, i) {
    return Object.assign({ _i: i }, c);
  });

  /* ---- constraint filtering (apply_constraints) ------------------------- */

  function applyConstraints(rows, k) {
    let out = rows;

    // Physically impossible predictions go first. A negative VRAM prediction
    // would otherwise satisfy every budget and then rescale the memory
    // objective for every other candidate.
    out = out.filter(function (r) { return !r.implausible; });

    if (k.maxVramGb != null) {
      const usable = k.maxVramGb * (1 - CFG.vram_safety_margin);
      out = out.filter(function (r) {
        return r.pred_peak_vram_gb != null && r.pred_peak_vram_gb <= usable;
      });
    }
    if (k.maxCarbon != null) {
      out = out.filter(function (r) { return r.pred_carbon_kgco2eq <= k.maxCarbon; });
    }
    if (k.minAccuracy != null) {
      out = out.filter(function (r) { return r.pred_accuracy >= k.minAccuracy; });
    }
    if (k.maxTimeSeconds != null) {
      out = out.filter(function (r) { return r.pred_wall_clock_s <= k.maxTimeSeconds; });
    }
    if (k.minConfidence) {
      const floor = CONFIDENCE_ORDER.indexOf(k.minConfidence);
      out = out.filter(function (r) {
        return CONFIDENCE_ORDER.indexOf(r.confidence) >= floor;
      });
    }
    if (k.methods && k.methods.length) {
      out = out.filter(function (r) { return k.methods.indexOf(r.method) >= 0; });
    }
    if (k.backbones && k.backbones.length) {
      out = out.filter(function (r) { return k.backbones.indexOf(r.backbone) >= 0; });
    }
    return out;
  }

  /* ---- Pareto frontier (pareto_front) ----------------------------------- */

  // True where no other row is at least as good on every axis and strictly
  // better on at least one. Accuracy up; VRAM, carbon and time down.
  function paretoMask(rows) {
    const n = rows.length;
    const flags = new Array(n).fill(true);
    for (let i = 0; i < n; i++) {
      const a = rows[i];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const b = rows[j];
        const geq = b.pred_accuracy >= a.pred_accuracy &&
                    b.pred_peak_vram_gb <= a.pred_peak_vram_gb &&
                    b.pred_carbon_kgco2eq <= a.pred_carbon_kgco2eq &&
                    b.pred_wall_clock_s <= a.pred_wall_clock_s;
        const gt = b.pred_accuracy > a.pred_accuracy ||
                   b.pred_peak_vram_gb < a.pred_peak_vram_gb ||
                   b.pred_carbon_kgco2eq < a.pred_carbon_kgco2eq ||
                   b.pred_wall_clock_s < a.pred_wall_clock_s;
        if (geq && gt) { flags[i] = false; break; }
      }
    }
    return flags;
  }

  /* ---- GEI scoring (score_gei) ------------------------------------------ */

  // Min-max over the supplied set only. A degenerate range means there is no
  // trade-off left to score on that axis, so every candidate gets 1.0.
  function normalise(values, higherIsBetter) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i++) {
      if (values[i] < lo) lo = values[i];
      if (values[i] > hi) hi = values[i];
    }
    if (!(hi - lo > 1e-12)) return values.map(function () { return 1; });
    return values.map(function (v) {
      const s = (v - lo) / (hi - lo);
      return higherIsBetter ? s : 1 - s;
    });
  }

  function scoreGEI(rows, weights) {
    const sAcc = normalise(rows.map(function (r) { return r.pred_accuracy; }), true);
    const sMem = normalise(rows.map(function (r) { return r.pred_peak_vram_gb; }), false);
    const sCar = normalise(rows.map(function (r) { return r.pred_carbon_kgco2eq; }), false);
    const sTim = normalise(rows.map(function (r) { return r.pred_wall_clock_s; }), false);
    const w = weights;

    return rows.map(function (r, i) {
      const parts = {
        accuracy: w[0] * sAcc[i],
        memory: w[1] * sMem[i],
        carbon: w[2] * sCar[i],
        time: w[3] * sTim[i]
      };
      return Object.assign({}, r, {
        S_Acc: sAcc[i], S_Mem: sMem[i], S_C: sCar[i], S_T: sTim[i],
        gei_parts: parts,
        gei: parts.accuracy + parts.memory + parts.carbon + parts.time
      });
    });
  }

  /* ---- scope tally (confidence.summarize) -------------------------------- */

  function scopeSummary(rows) {
    const counts = {};
    rows.forEach(function (r) { counts[r.scope] = (counts[r.scope] || 0) + 1; });
    return SCOPE_ORDER.slice().reverse()
      .filter(function (s) { return counts[s]; })
      .map(function (s) { return counts[s] + ' ' + s; })
      .join(', ');
  }

  /* ---- why nothing survived (recommend's dropped_summary) ---------------- */

  function dropReasons(pool, k) {
    const reasons = [];
    const total = pool.length;
    const implausible = pool.filter(function (r) { return r.implausible; }).length;

    if (implausible) {
      reasons.push(implausible + ' of ' + total + ' produced physically impossible predictions ' +
        '(negative VRAM, energy or time, or accuracy above 1.0) — the linear surrogates ' +
        'extrapolated outside the measured range.');
    }
    if (k.maxVramGb != null) {
      const usable = k.maxVramGb * (1 - CFG.vram_safety_margin);
      const over = pool.filter(function (r) { return r.pred_peak_vram_gb > usable; }).length;
      reasons.push(over + ' of ' + total + ' predicted above ' + usable.toFixed(1) +
        ' GB usable VRAM (' + k.maxVramGb + ' GB budget minus a ' +
        Math.round(CFG.vram_safety_margin * 100) + '% safety margin).');
    }
    if (k.minAccuracy != null) {
      const under = pool.filter(function (r) { return r.pred_accuracy < k.minAccuracy; }).length;
      reasons.push(under + ' of ' + total + ' predicted below the ' +
        k.minAccuracy.toFixed(3) + ' accuracy floor.');
    }
    if (k.maxCarbon != null) {
      const over = pool.filter(function (r) { return r.pred_carbon_kgco2eq > k.maxCarbon; }).length;
      reasons.push(over + ' of ' + total + ' predicted above the ' +
        k.maxCarbon.toFixed(4) + ' kgCO₂eq carbon budget.');
    }
    if (k.maxTimeSeconds != null) {
      const over = pool.filter(function (r) { return r.pred_wall_clock_s > k.maxTimeSeconds; }).length;
      reasons.push(over + ' of ' + total + ' predicted above the ' +
        Math.round(k.maxTimeSeconds) + ' s runtime limit.');
    }
    if (k.minConfidence) {
      const floor = CONFIDENCE_ORDER.indexOf(k.minConfidence);
      const below = pool.filter(function (r) {
        return CONFIDENCE_ORDER.indexOf(r.confidence) < floor;
      }).length;
      reasons.push(below + ' of ' + total + ' fell below the ' + k.minConfidence +
        ' evidence floor (outside the measured envelope).');
    }
    return reasons;
  }

  /* ---- top-level recommend() -------------------------------------------- */

  function recommend(k) {
    const profile = k.profile || 'balanced';
    const weights = k.weights || CFG.gei_profiles[profile];

    // The catalogue the user chose to consider, before any constraint applies.
    // Reasons are reported against this pool so the counts stay truthful when
    // methods or backbones have been deselected.
    let pool = ALL;
    if (k.methods && k.methods.length) {
      pool = pool.filter(function (r) { return k.methods.indexOf(r.method) >= 0; });
    }
    if (k.backbones && k.backbones.length) {
      pool = pool.filter(function (r) { return k.backbones.indexOf(r.backbone) >= 0; });
    }

    const feasible = applyConstraints(pool, k);

    const result = {
      profile: profile,
      weights: weights,
      constraints: k,
      nCatalogue: ALL.length,
      nPool: pool.length,
      nFeasible: feasible.length,
      nImplausible: pool.filter(function (r) { return r.implausible; }).length,
      modelStatus: DATA.model_status,
      scopeSummary: scopeSummary(feasible),
      pool: pool,
      feasible: [],
      ranked: [],
      nPareto: 0,
      reasons: []
    };

    if (!feasible.length) {
      result.reasons = dropReasons(pool, k);
      return result;
    }

    const front = paretoMask(feasible);
    const scored = scoreGEI(feasible, weights).map(function (r, i) {
      return Object.assign(r, { on_pareto_front: front[i] });
    });
    scored.sort(function (a, b) { return b.gei - a.gei; });

    result.feasible = scored;
    result.ranked = scored;
    result.nPareto = front.filter(Boolean).length;
    return result;
  }

  /* ---- catalogue helpers for the UI -------------------------------------- */

  const backbones = (function () {
    const seen = {}, list = [];
    ALL.forEach(function (r) {
      if (!seen[r.backbone]) {
        seen[r.backbone] = true;
        list.push({ key: r.backbone, model_id: r.model_id, params_b: r.params_b, family: r.family });
      }
    });
    return list.sort(function (a, b) { return a.params_b - b.params_b; });
  })();

  const methods = (function () {
    const seen = {}, list = [];
    ALL.forEach(function (r) { if (!seen[r.method]) { seen[r.method] = true; list.push(r.method); } });
    return list;
  })();

  return {
    data: DATA,
    config: CFG,
    all: ALL,
    backbones: backbones,
    methods: methods,
    methodLabel: function (m) { return DATA.method_labels[m] || m; },
    recommend: recommend,
    paretoMask: paretoMask,
    normalise: normalise,
    scopeSummary: scopeSummary,
    SCOPE_ORDER: SCOPE_ORDER,
    CONFIDENCE_ORDER: CONFIDENCE_ORDER
  };
})();
