/* ===========================================================================
   GreenPEFT showcase — page behaviour.

   Everything rendered here is driven by data/candidates.js and data/evidence.js,
   both written by analysis/build_website_data.py from the repository artifacts.
   No number in this file is typed by hand.
   ========================================================================= */

(function () {
  'use strict';

  const E = GPCharts.el;
  const EV = window.GREENPEFT_EVIDENCE;
  const CFG = GPEngine.config;
  const REPO = 'https://github.com/ai-tanzil811/Green-AI-Decision-Support-Frameworks';
  const BLOB = REPO + '/blob/main/';
  const TREE = REPO + '/tree/main/';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = function (sel, root) { return (root || document).querySelector(sel); };
  const $$ = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  /* ---------------------------------------------------- formatting ------- */

  const fmt = {
    acc: function (v) { return v.toFixed(4); },
    acc3: function (v) { return v.toFixed(3); },
    gb: function (v) { return v.toFixed(2); },
    gb1: function (v) { return v.toFixed(1); },
    kwh: function (v) { return v.toFixed(5); },
    kg: function (v) { return v.toFixed(5); },
    sec: function (v) { return v.toFixed(1); },
    gei: function (v) { return v.toFixed(4); },
    pct: function (v) { return v.toFixed(1) + '%'; },
    sci: function (v) { return v.toExponential(2); }
  };

  const METHOD_SHORT = {
    full_ft: 'Full FT', lora: 'LoRA', qlora: 'QLoRA', lora_fa: 'LoRA-FA', lisa: 'LISA'
  };
  function mLabel(m) { return METHOD_SHORT[m] || m; }

  const SCOPE_WORD = {
    MEASURED: 'measured directly',
    INTERPOLATED_SCALE: 'interpolated scale',
    UNSEEN_METHOD_SCALE: 'unmeasured pairing',
    UNSEEN_FAMILY: 'unseen family',
    OUT_OF_RANGE_SCALE: 'outside measured range'
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function toast(message) {
    const t = $('#toast');
    t.textContent = message;
    t.classList.add('is-on');
    clearTimeout(toast._t);
    toast._t = setTimeout(function () { t.classList.remove('is-on'); }, 1900);
  }

  function copy(text, message) {
    const done = function () { toast(message || 'Copied'); };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () { fallback(text, done); });
    } else {
      fallback(text, done);
    }
  }
  function fallback(text, done) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); done(); } catch (e) { toast('Copy failed'); }
    document.body.removeChild(ta);
  }

  /* ================================================== 1. the lab ========= */

  const state = {
    vram: 16,
    accuracy: 0.90,
    carbonOn: false,
    carbon: 0.003,
    timeOn: false,
    time: 300,
    profile: 'balanced',
    minConfidence: '',
    methods: GPEngine.methods.slice(),
    backbones: GPEngine.backbones.map(function (b) { return b.key; })
  };
  const DEFAULTS = JSON.parse(JSON.stringify(state));

  function constraints() {
    return {
      maxVramGb: state.vram,
      minAccuracy: state.accuracy,
      maxCarbon: state.carbonOn ? state.carbon : null,
      maxTimeSeconds: state.timeOn ? state.time : null,
      minConfidence: state.minConfidence || null,
      profile: state.profile,
      methods: state.methods,
      backbones: state.backbones
    };
  }

  /* ---- controls ---------------------------------------------------------- */

  function buildChipGroups() {
    const mWrap = $('#c-methods');
    GPEngine.methods.forEach(function (m) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.setAttribute('aria-pressed', 'true');
      b.dataset.method = m;
      b.innerHTML = '<span class="chip__dot"></span>' + escapeHtml(mLabel(m));
      b.addEventListener('click', function () {
        toggleIn(state.methods, m, b);
        runLab();
      });
      mWrap.appendChild(b);
    });

    const bWrap = $('#c-backbones');
    GPEngine.backbones.forEach(function (bb) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.setAttribute('aria-pressed', 'true');
      b.dataset.backbone = bb.key;
      b.title = bb.model_id + ' · ' + bb.params_b + 'B · ' + bb.family;
      b.innerHTML = '<span class="chip__dot"></span>' + escapeHtml(bb.key.replace(/_/g, '-')) +
                    ' <span class="faint">' + bb.params_b + 'B</span>';
      b.addEventListener('click', function () {
        toggleIn(state.backbones, bb.key, b);
        runLab();
      });
      bWrap.appendChild(b);
    });
  }

  function toggleIn(list, value, button) {
    const i = list.indexOf(value);
    if (i >= 0) {
      if (list.length === 1) { toast('Keep at least one'); return; }
      list.splice(i, 1);
      button.setAttribute('aria-pressed', 'false');
    } else {
      list.push(value);
      button.setAttribute('aria-pressed', 'true');
    }
  }

  function syncControls() {
    $('#c-vram').value = state.vram;
    $('#c-vram-out').innerHTML = state.vram + '<span class="u">GB</span>';
    $('#c-vram-hint').textContent =
      'Feasibility uses ' + (state.vram * (1 - CFG.vram_safety_margin)).toFixed(1) +
      ' GB usable — the budget minus a ' + Math.round(CFG.vram_safety_margin * 100) +
      '% safety margin, because the VRAM surrogate is only preliminary.';

    $('#c-acc').value = state.accuracy;
    $('#c-acc-out').textContent = state.accuracy.toFixed(3);

    $('#c-carbon-on').checked = state.carbonOn;
    $('#c-carbon').disabled = !state.carbonOn;
    $('#c-carbon').value = state.carbon;
    $('#c-carbon-out').innerHTML = state.carbonOn
      ? state.carbon.toFixed(4) + '<span class="u">kg</span>' : 'off';

    $('#c-time-on').checked = state.timeOn;
    $('#c-time').disabled = !state.timeOn;
    $('#c-time').value = state.time;
    $('#c-time-out').innerHTML = state.timeOn
      ? state.time + '<span class="u">s</span>' : 'off';

    $$('#c-profile button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.profile === state.profile));
    });
    const w = CFG.gei_profiles[state.profile];
    $('#c-profile-hint').textContent =
      'w = [' + w.join(', ') + '] over accuracy, memory, carbon, time.';

    $$('#c-conf button').forEach(function (b) {
      b.setAttribute('aria-pressed', String((b.dataset.conf || '') === state.minConfidence));
    });
    $('#c-conf-hint').textContent = {
      '': 'Every catalogue entry is considered; each one reports its own scope.',
      MEDIUM: 'Only candidates inside the measured parameter range and model families.',
      HIGH: 'Only the 14 (method, scale) cells that were benchmarked directly.'
    }[state.minConfidence];

    $$('#c-methods button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(state.methods.indexOf(b.dataset.method) >= 0));
    });
    $$('#c-backbones button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(state.backbones.indexOf(b.dataset.backbone) >= 0));
    });
  }

  function wireControls() {
    $('#c-vram').addEventListener('input', function (e) {
      state.vram = Number(e.target.value); runLab();
    });
    $('#c-acc').addEventListener('input', function (e) {
      state.accuracy = Number(e.target.value); runLab();
    });
    $('#c-carbon-on').addEventListener('change', function (e) {
      state.carbonOn = e.target.checked; runLab();
    });
    $('#c-carbon').addEventListener('input', function (e) {
      state.carbon = Number(e.target.value); runLab();
    });
    $('#c-time-on').addEventListener('change', function (e) {
      state.timeOn = e.target.checked; runLab();
    });
    $('#c-time').addEventListener('input', function (e) {
      state.time = Number(e.target.value); runLab();
    });
    $$('#c-profile button').forEach(function (b) {
      b.addEventListener('click', function () { state.profile = b.dataset.profile; runLab(); });
    });
    $$('#c-conf button').forEach(function (b) {
      b.addEventListener('click', function () { state.minConfidence = b.dataset.conf || ''; runLab(); });
    });
    $('#method-all').addEventListener('click', function () {
      state.methods = GPEngine.methods.slice(); runLab();
    });
    $('#bb-all').addEventListener('click', function () {
      state.backbones = GPEngine.backbones.map(function (b) { return b.key; }); runLab();
    });
    $('#reset-btn').addEventListener('click', function () {
      Object.assign(state, JSON.parse(JSON.stringify(DEFAULTS)));
      runLab();
      toast('Reset to defaults');
    });
    $('#lab-form').addEventListener('submit', function (e) { e.preventDefault(); });
  }

  /* ---- rendering the result --------------------------------------------- */

  function metric(label, value, unit, band) {
    return '<div class="metric">' +
      '<span class="metric__label">' + label + '</span>' +
      '<span class="metric__value">' + value +
        (unit ? '<span class="u">' + unit + '</span>' : '') + '</span>' +
      (band ? '<span class="metric__band">' + band + '</span>' : '') +
      '</div>';
  }

  function bandText(value, pct, format, unit) {
    if (pct == null) return '';
    const half = Math.abs(value) * pct / 100;
    return '±' + Math.round(pct) + '%  ' + format(value - half) + '–' + format(value + half) +
           (unit ? ' ' + unit : '');
  }

  function explain(res) {
    const top = res.ranked[0];
    const w = res.weights;
    const bits = [];

    const driver = ['accuracy', 'memory', 'carbon', 'time'][
      w.indexOf(Math.max.apply(null, w))
    ];
    bits.push('Under the <strong>' + res.profile.replace('_', ' ') + '</strong> profile, ' +
      driver + ' carries the heaviest weight (' + Math.max.apply(null, w) + ').');

    const parts = top.gei_parts;
    const ordered = Object.keys(parts).sort(function (a, b) { return parts[b] - parts[a]; });
    bits.push('<strong>' + mLabel(top.method) + ' on ' + escapeHtml(top.backbone.replace(/_/g, '-')) +
      '</strong> scores ' + fmt.gei(top.gei) + ', and most of that comes from its ' +
      ordered[0] + ' term (' + parts[ordered[0]].toFixed(3) + ') followed by ' +
      ordered[1] + ' (' + parts[ordered[1]].toFixed(3) + ').');

    bits.push('It uses <strong>' + fmt.gb(top.pred_peak_vram_gb) + ' GB</strong> of the ' +
      (state.vram * (1 - CFG.vram_safety_margin)).toFixed(1) + ' GB usable budget and clears the ' +
      state.accuracy.toFixed(3) + ' accuracy floor by ' +
      (top.pred_accuracy - state.accuracy).toFixed(4) + '.');

    bits.push(top.on_pareto_front
      ? 'No other feasible candidate beats it on accuracy, VRAM, carbon and time at once — it is on the Pareto front.'
      : 'It is <strong>not</strong> on the Pareto front: another feasible candidate matches or beats it on every objective, but the weights still favour this one. Check the alternatives below.');

    return bits.join(' ');
  }

  function renderRecommendation(res) {
    const panel = $('#rec-panel');

    if (!res.ranked.length) {
      panel.className = 'panel rec rec--none';
      panel.innerHTML =
        '<div class="rec__top">' +
          '<span class="badge badge--bad">Infeasible</span>' +
          '<h3 class="rec__method mt-sm">No candidate satisfies every constraint.</h3>' +
          '<p class="muted mt-sm" style="font-size:var(--fs-sm)">' +
            'Nothing survived out of ' + res.nPool + ' candidate' + (res.nPool === 1 ? '' : 's') +
            ' in the selected catalogue. This is a result, not an error — here is exactly what ruled each one out.' +
          '</p>' +
          '<ul class="reasons">' + res.reasons.map(function (r) {
            return '<li><span>' + escapeHtml(r) + '</span></li>';
          }).join('') + '</ul>' +
          '<p class="muted mt-md" style="font-size:var(--fs-xs)">' +
            'Loosen one constraint — raise the VRAM budget, lower the accuracy floor, or drop the evidence floor — and try again.' +
          '</p>' +
        '</div>';
      $('#alts').innerHTML = '<div class="alt"><span class="alt__name faint">No alternatives to show.</span></div>';
      $('#alts-note').textContent = '';
      $('#sbars').innerHTML = '<p class="chart-card__note">Nothing feasible to score.</p>';
      return;
    }

    const top = res.ranked[0];
    const confClass = { HIGH: 'ok', MEDIUM: 'warn', LOW: 'bad' }[top.confidence];

    panel.className = 'panel rec';
    panel.innerHTML =
      '<div class="rec__top">' +
        '<span class="kicker">Recommended configuration</span>' +
        '<div class="rec__id mt-sm">' +
          '<span class="rec__method">' + escapeHtml(mLabel(top.method)) + '</span>' +
          '<span class="rec__on">on</span>' +
          '<span class="rec__method">' + escapeHtml(top.backbone.replace(/_/g, '-')) + '</span>' +
          '<span class="rec__on mono">' + top.params_b + ' B · ' + escapeHtml(top.family) + '</span>' +
        '</div>' +
        '<div class="rec__model">' + escapeHtml(top.model_id) + '</div>' +
        '<div class="rec__badges">' +
          '<span class="badge badge--ok">Feasible</span>' +
          (top.on_pareto_front
            ? '<span class="badge badge--ok">Pareto-optimal</span>'
            : '<span class="badge badge--mute">Dominated</span>') +
          '<span class="badge badge--' + confClass + '">' + top.confidence + ' confidence</span>' +
          '<span class="badge badge--mute" title="' + escapeHtml(top.scope_reason) + '">' +
            escapeHtml(SCOPE_WORD[top.scope] || top.scope) + '</span>' +
          (top.quant_bits !== 16 ? '<span class="badge badge--mute">' + top.quant_bits + '-bit</span>' : '') +
          (top.rank ? '<span class="badge badge--mute">rank ' + top.rank + '</span>' : '') +
        '</div>' +
      '</div>' +

      '<div class="rec__metrics">' +
        metric('GEI score', fmt.gei(top.gei), '',
          'rank 1 of ' + res.nFeasible + ' feasible') +
        metric('Predicted accuracy', fmt.acc(top.pred_accuracy), '',
          bandText(top.pred_accuracy, top.pred_accuracy_band_pct, fmt.acc)) +
        metric('Peak VRAM', fmt.gb(top.pred_peak_vram_gb), 'GB',
          bandText(top.pred_peak_vram_gb, top.pred_peak_vram_gb_band_pct, fmt.gb1, 'GB')) +
        metric('Energy', fmt.kwh(top.pred_energy_kwh), 'kWh',
          bandText(top.pred_energy_kwh, top.pred_energy_kwh_band_pct, fmt.kwh)) +
        metric('Carbon (kgCO₂eq)', fmt.kg(top.pred_carbon_kgco2eq), '',
          'derived as energy × ' + CFG.grid_carbon_kg_per_kwh + ', never modelled') +
        metric('Wall-clock', fmt.sec(top.pred_wall_clock_s), 's',
          bandText(top.pred_wall_clock_s, top.pred_wall_clock_s_band_pct, fmt.sec, 's')) +
      '</div>' +

      '<p class="rec__why">' + explain(res) + '</p>' +

      '<p class="rec__why">' +
        '<strong>' + res.nFeasible + ' of ' + res.nPool + '</strong> candidates satisfied every ' +
        'constraint; <strong>' + res.nPareto + '</strong> of those are Pareto-optimal' +
        (res.nImplausible ? ', and ' + res.nImplausible + ' were dropped before scoring for ' +
          'physically impossible predictions' : '') + '. ' +
        (res.scopeSummary ? 'Feasible set by evidence scope: <span class="mono">' +
          escapeHtml(res.scopeSummary.toLowerCase()) + '</span>. ' : '') +
        (top.scope_caveat ? '<br><span class="accent">Caveat:</span> ' + escapeHtml(top.scope_caveat) + '. ' : '') +
        'Surrogate status is <strong>' + res.modelStatus.toLowerCase() + '</strong> — treat this as a ' +
        'shortlist to validate, not an approval to skip measuring.' +
      '</p>' +

      '<div class="rec__why" style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">' +
        '<button type="button" class="btn btn--ghost btn--sm" id="copy-equiv">Copy equivalent CLI command</button>' +
        '<span class="faint" style="font-size:var(--fs-label)">reproduces this exact query on your machine</span>' +
      '</div>';

    $('#copy-equiv').addEventListener('click', function () {
      copy(equivalentCommand(), 'CLI command copied');
    });
  }

  function equivalentCommand() {
    const parts = ['green-peft recommend', '--vram ' + state.vram,
                   '--accuracy ' + state.accuracy.toFixed(3)];
    if (state.carbonOn) parts.push('--carbon ' + state.carbon.toFixed(4));
    if (state.timeOn) parts.push('--time ' + state.time);
    parts.push('--profile ' + state.profile);
    if (state.minConfidence) parts.push('--min-confidence ' + state.minConfidence);
    if (state.methods.length < GPEngine.methods.length) {
      parts.push('--methods ' + state.methods.join(','));
    }
    if (state.backbones.length < GPEngine.backbones.length) {
      parts.push('--backbones ' + state.backbones.join(','));
    }
    return parts.join(' \\\n  ');
  }

  function renderAlternatives(res) {
    const wrap = $('#alts');
    const rows = res.ranked.slice(1, 6);
    $('#alts-note').textContent = res.nFeasible > 1
      ? 'ranks 2–' + Math.min(res.nFeasible, 6) + ' of ' + res.nFeasible
      : 'no runner-up';

    if (!rows.length) {
      wrap.innerHTML = '<div class="alt"><span class="alt__name faint">' +
        'Only one candidate is feasible under these constraints.</span></div>';
      return;
    }

    const max = res.ranked[0].gei || 1;
    wrap.innerHTML = rows.map(function (r, i) {
      return '<div class="alt">' +
        '<span class="alt__rank">' + (i + 2) + '</span>' +
        '<span>' +
          '<span class="alt__name"><b>' + escapeHtml(mLabel(r.method)) + '</b> ' +
            '<span>· ' + escapeHtml(r.backbone.replace(/_/g, '-')) + ' · ' + r.params_b + 'B</span></span>' +
          '<span class="alt__meta">' + fmt.acc(r.pred_accuracy) + ' acc · ' +
            fmt.gb(r.pred_peak_vram_gb) + ' GB · ' + fmt.kg(r.pred_carbon_kgco2eq) + ' kg · ' +
            fmt.sec(r.pred_wall_clock_s) + ' s · ' + r.confidence.toLowerCase() +
            (r.on_pareto_front ? ' · pareto' : '') + '</span>' +
        '</span>' +
        '<span class="alt__gei">' +
          '<span class="alt__bar"><i style="width:' + ((r.gei / max) * 100).toFixed(1) + '%"></i></span>' +
          '<span class="alt__score">' + fmt.gei(r.gei) + '</span>' +
        '</span>' +
      '</div>';
    }).join('');
  }

  function renderScoreBars(res) {
    const wrap = $('#sbars');
    const rows = res.ranked.slice(0, 5);
    if (!rows.length) { wrap.innerHTML = ''; return; }

    wrap.innerHTML = rows.map(function (r) {
      const p = r.gei_parts;
      const total = r.gei || 1;
      const seg = function (cls, v) {
        return '<span class="sbar__seg sbar__seg--' + cls + '" style="width:' +
          ((v / total) * 100).toFixed(2) + '%"></span>';
      };
      return '<div class="sbar">' +
        '<span class="sbar__head">' +
          '<span class="sbar__name">' + escapeHtml(mLabel(r.method)) + ' · ' +
            escapeHtml(r.backbone.replace(/_/g, '-')) + '</span>' +
          '<span class="alt__score">' + fmt.gei(r.gei) + '</span>' +
        '</span>' +
        '<span class="sbar__track" title="accuracy ' + p.accuracy.toFixed(3) +
          ' · memory ' + p.memory.toFixed(3) + ' · carbon ' + p.carbon.toFixed(3) +
          ' · time ' + p.time.toFixed(3) + '">' +
          seg('acc', p.accuracy) + seg('mem', p.memory) +
          seg('car', p.carbon) + seg('tim', p.time) +
        '</span>' +
      '</div>';
    }).join('');
  }

  function renderLabScatter(res) {
    const mount = $('#lab-scatter');
    mount.innerHTML = '';

    const feasibleKeys = {};
    res.feasible.forEach(function (r) { feasibleKeys[r.backbone + '|' + r.method] = r; });
    const topKey = res.ranked.length ? res.ranked[0].backbone + '|' + res.ranked[0].method : null;

    const points = res.pool
      .filter(function (r) { return !r.implausible && r.pred_peak_vram_gb > 0; })
      .map(function (r) {
        const key = r.backbone + '|' + r.method;
        const isTop = key === topKey;
        const isFeasible = !!feasibleKeys[key];
        return {
          x: r.pred_peak_vram_gb,
          y: r.pred_accuracy,
          shape: GPCharts.SHAPES[r.method] || 'circle',
          color: isTop ? '#d7a13f' : (isFeasible ? '#4fbdaf' : '#6e8378'),
          filled: isTop || isFeasible,
          size: isTop ? 6 : 4,
          ring: isTop ? '#f0d7a4' : null,
          title: mLabel(r.method) + ' · ' + r.backbone + ' (' + r.params_b + 'B)\n' +
                 fmt.gb(r.pred_peak_vram_gb) + ' GB · ' + fmt.acc(r.pred_accuracy) + ' acc · ' +
                 r.scope
        };
      });

    if (!points.length) {
      mount.innerHTML = '<p class="chart-card__note">No plausible candidate left to plot.</p>';
      return;
    }

    mount.appendChild(GPCharts.scatter({
      points: points,
      width: 500, height: 300,
      xLabel: 'predicted peak VRAM (GB)',
      yLabel: 'predicted accuracy',
      xFormat: function (v) { return v.toFixed(0); },
      yFormat: function (v) { return v.toFixed(2); },
      rules: [
        { axis: 'x', value: state.vram * (1 - CFG.vram_safety_margin),
          text: (state.vram * (1 - CFG.vram_safety_margin)).toFixed(1) + ' GB usable' },
        { axis: 'y', value: state.accuracy, text: 'floor ' + state.accuracy.toFixed(2) }
      ]
    }));
  }

  let lastResult = null;
  function runLab() {
    syncControls();
    const res = GPEngine.recommend(constraints());
    lastResult = res;
    renderRecommendation(res);
    renderAlternatives(res);
    renderScoreBars(res);
    renderLabScatter(res);
  }

  /* ================================================== 2. hero cascade ==== */

  function buildCascade() {
    const mount = $('#cascade-mount');
    const methods = GPEngine.methods;
    const backbones = GPEngine.backbones;
    const byKey = {};
    GPEngine.all.forEach(function (c) { byKey[c.backbone + '|' + c.method] = c; });

    // Default hero scenario: a 16 GB device, a 0.90 accuracy floor, balanced weights.
    const k = { maxVramGb: 16, minAccuracy: 0.90, profile: 'balanced' };
    const res = GPEngine.recommend(k);
    const usable = 16 * (1 - CFG.vram_safety_margin);

    const feasible = {};
    res.feasible.forEach(function (r) { feasible[r.backbone + '|' + r.method] = true; });
    const topKey = res.ranked.length ? res.ranked[0].backbone + '|' + res.ranked[0].method : null;

    const stateOf = {};
    GPEngine.all.forEach(function (c) {
      const key = c.backbone + '|' + c.method;
      if (c.implausible) stateOf[key] = 'bad';
      else if (c.pred_peak_vram_gb > usable) stateOf[key] = 'vram';
      else if (c.pred_accuracy < 0.90) stateOf[key] = 'acc';
      else stateOf[key] = 'pass';
    });

    const built = GPCharts.cascadeMatrix({
      methods: methods, backbones: backbones, byKey: byKey, state: stateOf
    });
    mount.appendChild(built.node);

    // Ranked strip under the matrix.
    const strip = document.createElement('div');
    strip.className = 'cascade__rank';
    strip.innerHTML = res.ranked.slice(0, 3).map(function (r, i) {
      return '<div class="crank">' +
        '<span class="crank__n">' + (i + 1) + '</span>' +
        '<span class="crank__name"><b>' + escapeHtml(mLabel(r.method)) + '</b> · ' +
          escapeHtml(r.backbone.replace(/_/g, '-')) + ' <span class="faint">' + r.params_b + 'B</span></span>' +
        '<span class="crank__bar"><i style="width:' +
          ((r.gei / res.ranked[0].gei) * 100).toFixed(1) + '%"></i></span>' +
        '<span class="crank__v">' + fmt.gei(r.gei) + '</span>' +
      '</div>';
    }).join('');
    mount.appendChild(strip);

    $('#cascade-foot').innerHTML =
      'Each cell is one candidate, labelled with its predicted peak VRAM in GB. ' +
      '<span class="accent">' + res.nFeasible + '</span> of ' + res.nCatalogue +
      ' clear the gates; ' + res.nImplausible + ' are dropped as physically impossible, ' +
      'and ' + res.nPareto + ' of the survivors are Pareto-optimal.';

    // Staged reveal — cells fade in gate by gate. Reduced motion jumps to the end.
    const cells = $$('.ccell', built.node);
    const order = { pass: 0, acc: 1, vram: 2, bad: 3 };
    if (reduceMotion) {
      cells.forEach(function (c) { c.setAttribute('opacity', 1); });
      strip.classList.add('is-in');
      return;
    }
    cells.forEach(function (c) {
      const st = stateOf[c.dataset.key] || 'other';
      const delay = 180 + order[st] * 340 + Math.random() * 180;
      setTimeout(function () { c.setAttribute('opacity', 1); }, delay);
    });
    setTimeout(function () { strip.classList.add('is-in'); }, 1500);
  }

  /* ================================================== 3. why / shift ===== */

  function buildShift() {
    const feas = EV.feasibility;
    const before = $('#shift-before');
    const after = $('#shift-after');

    // Row-major over the four backbone tiers × five strategies, matching the
    // benchmark grid: 20 cells, 60 attempted runs.
    const tiers = ['tiny', 'small', 'medium', 'large'];
    const methods = ['full_ft', 'lora', 'qlora', 'lora_fa', 'lisa'];
    const lookup = {};
    feas.forEach(function (f) { lookup[f.method + '|' + f.backbone] = f; });

    // The shortlist a 16 GB / 0.90 balanced query returns, restricted to
    // measured cells so the two panels compare like with like.
    const shortlist = {};
    GPEngine.recommend({
      maxVramGb: 16, minAccuracy: 0.90, profile: 'balanced', minConfidence: 'HIGH'
    }).ranked.slice(0, 3).forEach(function (r) {
      shortlist[r.method + '|' + r.backbone] = true;
    });

    let a = '', b = '';
    tiers.forEach(function (t) {
      methods.forEach(function (m) {
        const f = lookup[m + '|' + t] || { outcome: 'all_oom', runs_successful: 0 };
        const oom = f.outcome === 'all_oom';
        a += '<span class="gcell ' + (oom ? 'gcell--oom' : 'gcell--run') + '">' +
             (oom ? 'OOM' : f.runs_successful + '/3') + '</span>';
        const picked = shortlist[m + '|' + t];
        b += '<span class="gcell ' + (picked ? 'gcell--pick' : 'gcell--off') + '">' +
             (picked ? '✓' : '·') + '</span>';
      });
    });
    before.innerHTML = a;
    after.innerHTML = b;
  }

  /* ================================================== 4. pipeline ======== */

  const STAGES = [
    {
      title: 'Multi-source data harmonisation',
      in: ['surrogate_dataset.csv', 'Open LLM-Perf', 'llmenergy.csv'],
      out: ['477 × 46 ML-ready table', 'measurement-pass flags']
    },
    {
      title: 'Empirical benchmarking',
      in: ['backbones 0.5–3.0B', '5 PEFT strategies', 'seeds 13 · 42 · 2024'],
      out: ['60 runs on a Tesla T4', 'VRAM · energy · acc · time']
    },
    {
      title: 'Zero-shot surrogate regression',
      in: ['params_b · rank', 'quant_bits · dataset size', 'family · method'],
      out: ['4 Ridge regressors', 'per-target error bands', 'scope + confidence']
    },
    {
      title: 'Constraint filtering and Pareto optimisation',
      in: ['VRAM · accuracy budgets', 'carbon · time budgets', 'evidence floor'],
      out: ['feasible set', 'non-dominated frontier', 'GEI ranking']
    },
    {
      title: 'CLI recommendation and back-testing',
      in: ['green-peft recommend', 'recorded real runs'],
      out: ['ranked shortlist', 'predicted vs actual log']
    }
  ];

  function buildPipeline() {
    const mount = $('#pipeline-mount');
    const built = GPCharts.pipeline(STAGES);
    built.node.style.minWidth = built.width + 'px';
    const scroller = document.createElement('div');
    scroller.className = 'tbl-scroll';
    scroller.appendChild(built.node);
    mount.appendChild(scroller);
  }

  /* ================================================== 5. evidence ======== */

  function buildBenchStats() {
    const m = EV.manifest;
    const oom = EV.feasibility.reduce(function (n, f) { return n + f.runs_oom; }, 0);
    const stats = [
      [m.n_raw_runs, 'attempted runs'],
      [m.gpu, 'single GPU, 15.6 GB usable'],
      ['SST-2', m.train_steps + ' steps · batch ' + m.batch_size],
      [m.seeds.length + ' seeds', m.seeds.join(' · ')],
      [m.methods.length + ' strategies', m.methods.map(mLabel).join(' · ')],
      [oom, 'out-of-memory failures']
    ];
    $('#bench-stats').innerHTML = stats.map(function (s) {
      return '<div class="metric">' +
        '<span class="metric__value">' + escapeHtml(String(s[0])) + '</span>' +
        '<span class="metric__label">' + escapeHtml(s[1]) + '</span>' +
        '</div>';
    }).join('');
  }

  function measuredPoints(xKey, yKey) {
    return EV.measured.map(function (r) {
      return {
        x: r[xKey], y: r[yKey],
        shape: GPCharts.SHAPES[r.method] || 'circle',
        color: r.pareto_optimal ? '#1e6f66' : '#6f7a73',
        filled: !!r.pareto_optimal,
        size: 5,
        title: mLabel(r.method) + ' · ' + r.backbone + ' (' + r.params_b + 'B, ' + r.family + ')\n' +
               'accuracy ' + fmt.acc(r.accuracy) + '\n' +
               'peak VRAM ' + fmt.gb(r.peak_gpu_memory_gb) + ' GB\n' +
               'energy ' + fmt.kwh(r.energy_kwh) + ' kWh\n' +
               'wall-clock ' + fmt.sec(r.wall_clock_seconds) + ' s\n' +
               (r.pareto_optimal ? 'non-dominated' : 'dominated') + ' · ' + r.n_seeds + ' seeds'
      };
    });
  }

  function buildEvidenceCharts() {
    const accVram = $('#chart-acc-vram');
    const pts = measuredPoints('peak_gpu_memory_gb', 'accuracy');

    // Name the two cells the worked example turns on.
    EV.measured.forEach(function (r, i) {
      if (r.method === 'lisa' && r.backbone === 'medium') {
        pts[i].label = 'LISA / 1.5B'; pts[i].labelSide = 'left';
      }
      if (r.method === 'full_ft') { pts[i].label = 'Full FT / 0.5B'; pts[i].labelSide = 'left'; }
    });

    accVram.appendChild(GPCharts.scatter({
      points: pts,
      width: 500, height: 320,
      xLabel: 'peak VRAM (GB)',
      yLabel: 'accuracy',
      xFormat: function (v) { return v.toFixed(0); },
      yFormat: function (v) { return v.toFixed(2); },
      rules: [{ axis: 'x', value: 15.6, text: '15.6 GB usable' }]
    }));

    $('#chart-energy-acc').appendChild(GPCharts.scatter({
      points: measuredPoints('energy_kwh', 'accuracy'),
      width: 500, height: 320,
      xLabel: 'training energy (kWh)',
      yLabel: 'accuracy',
      xFormat: function (v) { return v.toFixed(3); },
      yFormat: function (v) { return v.toFixed(2); }
    }));
  }

  function buildFeasibilityMatrix() {
    const table = $('#feasibility-matrix');
    // A <caption> must be the table's first child or the parser drops it.
    const caption = table.querySelector('caption').outerHTML;
    const tiers = ['tiny', 'small', 'medium', 'large'];
    const tierLabel = { tiny: 'tiny · 0.5B', small: 'small · 1.1B', medium: 'medium · 1.5B', large: 'large · 3.0B' };
    const methods = ['full_ft', 'lora', 'qlora', 'lora_fa', 'lisa'];
    const lookup = {};
    EV.feasibility.forEach(function (f) { lookup[f.method + '|' + f.backbone] = f; });

    let html = caption + '<thead><tr><th scope="col"></th>' +
      methods.map(function (m) { return '<th scope="col">' + escapeHtml(mLabel(m)) + '</th>'; }).join('') +
      '</tr></thead><tbody>';

    tiers.forEach(function (t) {
      html += '<tr><th scope="row">' + escapeHtml(tierLabel[t]) + '</th>';
      methods.forEach(function (m) {
        const f = lookup[m + '|' + t];
        if (!f) { html += '<td></td>'; return; }
        const cls = f.outcome === 'all_ok' ? 'ok' : (f.outcome === 'mixed' ? 'mixed' : 'oom');
        const star = (m === 'lisa' && t === 'medium') ? ' cell--star' : '';
        html += '<td><span class="cell cell--' + cls + star + '" title="' +
          escapeHtml(mLabel(m) + ' on ' + t + ': ' + f.runs_successful + '/' + f.runs_attempted +
            ' runs completed, OOM rate ' + f.oom_rate) + '">' +
          '<b>' + f.runs_successful + '/' + f.runs_attempted + '</b>' +
          '<span>' + (f.outcome === 'all_oom' ? 'OOM' : (f.outcome === 'mixed' ? 'partial' : 'ok')) + '</span>' +
          '</span></td>';
      });
      html += '</tr>';
    });
    table.innerHTML = html + '</tbody>';
  }

  const MEASURED_COLS = [
    { key: 'method', label: 'Strategy', fmt: function (v) { return mLabel(v); } },
    { key: 'backbone', label: 'Backbone' },
    { key: 'params_b', label: 'Params (B)', n: true, fmt: function (v) { return v.toFixed(1); } },
    { key: 'n_seeds', label: 'Seeds', n: true },
    { key: 'accuracy', label: 'Accuracy', n: true, fmt: fmt.acc },
    { key: 'peak_gpu_memory_gb', label: 'Peak VRAM (GB)', n: true, fmt: fmt.gb },
    { key: 'energy_kwh', label: 'Energy (kWh)', n: true, fmt: fmt.kwh },
    { key: 'carbon_kgco2eq', label: 'Carbon (kgCO₂eq)', n: true, fmt: fmt.kg },
    { key: 'wall_clock_seconds', label: 'Wall-clock (s)', n: true, fmt: fmt.sec },
    { key: 'gei_balanced', label: 'GEI balanced', n: true, fmt: fmt.gei },
    { key: 'pareto_optimal', label: 'Pareto', fmt: function (v) { return v ? 'yes' : '—'; } }
  ];

  function buildMeasuredTable() {
    const table = $('#measured-table');
    const caption = table.querySelector('caption').outerHTML;
    let sortKey = 'gei_balanced', sortDir = -1;

    function draw() {
      const rows = EV.measured.slice().sort(function (a, b) {
        const x = a[sortKey], y = b[sortKey];
        if (typeof x === 'string') return sortDir * x.localeCompare(y);
        return sortDir * ((x > y) - (x < y));
      });

      table.innerHTML = caption +
        '<thead><tr>' + MEASURED_COLS.map(function (c) {
          const active = c.key === sortKey;
          return '<th scope="col" class="' + (c.n ? 'n' : '') + '">' +
            '<button type="button" data-key="' + c.key + '" style="font:inherit;letter-spacing:inherit;' +
            'text-transform:inherit;color:' + (active ? 'var(--accent)' : 'inherit') + '">' +
            escapeHtml(c.label) + (active ? (sortDir < 0 ? ' ↓' : ' ↑') : '') + '</button></th>';
        }).join('') + '</tr></thead>' +
        '<tbody>' + rows.map(function (r) {
          return '<tr class="' + (r.pareto_optimal ? 'is-marked' : '') + '">' +
            MEASURED_COLS.map(function (c) {
              const v = r[c.key];
              return '<td class="' + (c.n ? 'n' : '') + '">' +
                escapeHtml(c.fmt ? c.fmt(v) : String(v)) + '</td>';
            }).join('') + '</tr>';
        }).join('') + '</tbody>';

      $$('th button', table).forEach(function (b) {
        b.addEventListener('click', function () {
          if (sortKey === b.dataset.key) sortDir = -sortDir;
          else { sortKey = b.dataset.key; sortDir = -1; }
          draw();
        });
      });
    }
    draw();
  }

  function buildFigures() {
    $('#figstrip').innerHTML = EV.figures.map(function (f) {
      return '<figure class="fig">' +
        '<a class="fig__frame" href="' + escapeHtml(f.file) + '" target="_blank" rel="noopener">' +
          '<img src="' + escapeHtml(f.file) + '" alt="' + escapeHtml(f.caption) + '" loading="lazy" decoding="async">' +
        '</a>' +
        '<figcaption>' + escapeHtml(f.caption) +
          '<span class="fig__src">' + escapeHtml(f.source) + '</span>' +
        '</figcaption>' +
      '</figure>';
    }).join('');
  }

  /* ================================================== 6. GEI section ===== */

  const WEIGHT_META = [
    { name: 'Accuracy', colour: '#4fbdaf' },
    { name: 'Memory', colour: '#78b0d8' },
    { name: 'Carbon', colour: '#d7a13f' },
    { name: 'Time', colour: '#b08fd0' }
  ];

  function geiRanking(profile) {
    const key = 'gei_' + profile;
    return EV.measured.slice()
      .map(function (r) { return { r: r, v: r[key] }; })
      .sort(function (a, b) { return b.v - a.v; });
  }

  const BASELINE = geiRanking('balanced').map(function (x) {
    return x.r.method + '|' + x.r.backbone;
  });

  function renderGei(profile) {
    const w = CFG.gei_profiles[profile];

    $('#gei-weights').innerHTML = WEIGHT_META.map(function (m, i) {
      return '<div class="wbar">' +
        '<span class="wbar__head">' +
          '<span class="wbar__name">' + m.name + '</span>' +
          '<span class="wbar__val">' + w[i].toFixed(2) + '</span>' +
        '</span>' +
        '<span class="wbar__track"><span class="wbar__fill" style="width:' +
          (w[i] * 100).toFixed(1) + '%;background:' + m.colour + '"></span></span>' +
      '</div>';
    }).join('');

    const ranked = geiRanking(profile);
    const max = ranked[0].v;
    $('#gei-rank-note').textContent = profile.replace('_', ' ');
    $('#gei-rank').innerHTML = ranked.map(function (x, i) {
      const key = x.r.method + '|' + x.r.backbone;
      const move = BASELINE.indexOf(key) - i;
      const delta = move === 0 ? '<span class="rank__delta faint">·</span>'
        : '<span class="rank__delta rank__delta--' + (move > 0 ? 'up' : 'down') + '">' +
          (move > 0 ? '▲' : '▼') + Math.abs(move) + '</span>';
      return '<div class="rank__row' + (i === 0 ? ' is-top' : '') + '">' +
        '<span class="rank__n">' + (i + 1) + '</span>' +
        '<span class="rank__name"><b>' + escapeHtml(mLabel(x.r.method)) + '</b> ' +
          '<span>· ' + escapeHtml(x.r.backbone) + ' · ' + x.r.params_b + 'B</span></span>' +
        '<span class="rank__right">' +
          '<span class="rank__bar"><i style="width:' + ((x.v / max) * 100).toFixed(1) + '%"></i></span>' +
          '<span class="rank__v">' + x.v.toFixed(3) + '</span>' + delta +
        '</span>' +
      '</div>';
    }).join('');
  }

  function wireGei() {
    renderGei('balanced');
    $$('#gei-profile button').forEach(function (b) {
      b.addEventListener('click', function () {
        $$('#gei-profile button').forEach(function (o) {
          o.setAttribute('aria-pressed', String(o === b));
        });
        renderGei(b.dataset.profile);
      });
    });
  }

  /* ================================================== 7. scope =========== */

  const TARGET_LABEL = {
    accuracy: 'Accuracy',
    peak_gpu_memory_gb: 'Peak VRAM',
    energy_kwh: 'Energy',
    wall_clock_seconds: 'Wall-clock'
  };

  function buildCvTable() {
    const rows = EV.cv_metrics;
    const status = GPEngine.data.status_per_target;

    const cvCaption =
      '<caption>Governing status per target is taken from the weaker protocol: ' +
      Object.keys(status).map(function (k) {
        return escapeHtml(TARGET_LABEL[k]) + ' ' + status[k].toLowerCase();
      }).join(', ') + '. Source: <span class="mono">results/surrogate/cv_metrics.csv</span>.</caption>';

    $('#cv-table').innerHTML = cvCaption +
      '<thead><tr><th scope="col">Target</th><th scope="col">Protocol</th>' +
      '<th scope="col" class="n">R²</th><th scope="col" class="n">MAPE</th>' +
      '<th scope="col">Status</th></tr></thead><tbody>' +
      rows.map(function (r) {
        const grouped = r.protocol.indexOf('GroupKFold') === 0;
        const ok = grouped ? (r.R2 >= 0.7 && r.MAPE_pct <= 15) : (r.R2 >= 0.7 && r.MAPE_pct <= 15);
        return '<tr>' +
          '<td>' + escapeHtml(TARGET_LABEL[r.target] || r.target) + '</td>' +
          '<td class="faint" style="font-size:var(--fs-label)">' +
            (grouped ? 'GroupKFold · config' : 'Leave-one-tier-out') + '</td>' +
          '<td class="n">' + r.R2.toFixed(3) + '</td>' +
          '<td class="n">' + r.MAPE_pct.toFixed(2) + '%</td>' +
          '<td><span class="badge badge--' + (ok ? 'ok' : 'warn') + '">' +
            (ok ? 'validated' : 'preliminary') + '</span></td>' +
        '</tr>';
      }).join('') + '</tbody>';
  }

  function buildBacktestTable() {
    $('#backtest-table').innerHTML =
      '<caption>Predicted versus measured across ' + EV.backtest_n_configs +
      ' recorded configurations, all in-sample. Source: ' +
      '<span class="mono">results/recommendation_validation.csv</span>.</caption>' +
      '<thead><tr><th scope="col">Target</th><th scope="col" class="n">Mean |error|</th>' +
      '<th scope="col" class="n">Median |error|</th><th scope="col" class="n">Within band</th></tr></thead><tbody>' +
      EV.backtest.map(function (r) {
        return '<tr>' +
          '<td>' + escapeHtml(TARGET_LABEL[r.target] || r.target) + '</td>' +
          '<td class="n">' + r.mean_abs_pct_error.toFixed(2) + '%</td>' +
          '<td class="n">' + r.median_abs_pct_error.toFixed(2) + '%</td>' +
          '<td class="n">' + Math.round(r.within_band * 100) + '%</td>' +
        '</tr>';
      }).join('') + '</tbody>';
  }

  function buildEnvelopeStats() {
    const env = GPEngine.data.envelope;
    const stats = [
      [env.params_b_min + '–' + env.params_b_max + ' B', 'measured parameter span'],
      [env.measured_cells.length + ' / 20', 'directly measured cells'],
      [env.n_configs, 'unique configurations fitted'],
      [env.families.join(' · '), 'model families seen'],
      [env.gpu_total_gb + ' GB', 'usable device budget'],
      [env.task, 'the only benchmarked task']
    ];
    $('#envelope-stats').innerHTML = stats.map(function (s) {
      return '<div class="metric">' +
        '<span class="metric__value" style="font-size:1.05rem">' + escapeHtml(String(s[0])) + '</span>' +
        '<span class="metric__label">' + escapeHtml(s[1]) + '</span>' +
        '</div>';
    }).join('');
  }

  /* ================================================== 8. CLI ============= */

  const CLI = {
    recommend: {
      cmd: [
        ['$ ', 'pip install', ' green-peft'],
        null,
        ['$ ', 'green-peft recommend', ' \\'],
        ['', '', '    --artifacts-dir ./models/artifacts_export \\'],
        ['', '', '    --vram 16 \\'],
        ['', '', '    --accuracy 0.90 \\'],
        ['', '', '    --profile balanced']
      ],
      out: `Recommendation (balanced profile, weights=(0.35, 0.25, 0.25, 0.15)):
  qlora on medium_sl (HuggingFaceTB/SmolLM2-1.7B, 1.7B params)
  predicted accuracy   : 0.9421   +/-2%  [0.9269 to 0.9573]
  predicted peak VRAM  : 5.33 GB   +/-40%  [3.20 to 7.45 GB]
  predicted energy     : 0.003590 kWh   +/-11%  [0.003211 to 0.003970 kWh]
  predicted carbon     : 0.002334 kgCO2eq   derived as energy x 0.65, not modelled
  predicted wall-clock : 197.8 s   +/-23%  [152.3 to 243.2 s]
  GEI score            : 0.7938  [on Pareto front]
  evidence             : MEDIUM confidence, scope INTERPOLATED_SCALE
                         (1.7B between measured tiers)
  33/70 candidates satisfied all constraints, 25 of those are Pareto-optimal.
  feasible set by scope: 12 MEASURED, 10 INTERPOLATED_SCALE, 5 UNSEEN_FAMILY,
                         6 OUT_OF_RANGE_SCALE
  12/70 candidates were dropped before scoring for physically impossible
  predictions -- the linear surrogates extrapolate past those limits well
  outside the measured range.
  Surrogate status is PRELIMINARY: error bands above are out-of-sample
  measurements, not calibrated prediction intervals.

Top 3 by GEI:
 backbone method  pred_accuracy  pred_peak_vram_gb  pred_carbon    gei  pareto
medium_sl  qlora         0.9421             5.3290       0.0023 0.7938    True
large_phi  qlora         0.9499             6.9506       0.0034 0.7860    True
medium_q3  qlora         0.9424             5.9589       0.0024 0.7794    True`
    },
    carbon: {
      cmd: [
        ['$ ', 'green-peft recommend', ' \\'],
        ['', '', '    --vram 16 \\'],
        ['', '', '    --carbon 0.003 \\'],
        ['', '', '    --accuracy 0.90 \\'],
        ['', '', '    --profile strict_carbon \\'],
        ['', '', '    --min-confidence HIGH']
      ],
      out: `Restricting to directly measured cells under a carbon cap leaves a much
smaller, much better evidenced feasible set. The engine reports the scope of
every survivor rather than hiding the difference:

  feasible set by scope: 12 MEASURED

Lowering --min-confidence to MEDIUM or dropping it entirely widens the search
to interpolated scales and unseen families — the surrogate will still answer,
but it labels each candidate LOW or MEDIUM confidence and attaches the
leave-one-tier-out error band instead of the GroupKFold one.`
    },
    zoo: {
      cmd: [
        ['$ ', 'green-peft list-zoo', ' \\'],
        ['', '', '    --artifacts-dir ./models/artifacts_export']
      ],
      out: ` backbone                           model_id   family  params_b  method  confidence               scope
     tiny                    Qwen/Qwen2.5-0.5B     qwen     0.500 full_ft        HIGH            MEASURED
     tiny                    Qwen/Qwen2.5-0.5B     qwen     0.500    lora        HIGH            MEASURED
    small   TinyLlama/TinyLlama-1.1B-Chat-v1.0    llama     1.100    lisa        HIGH            MEASURED
   medium                    Qwen/Qwen2.5-1.5B     qwen     1.500 lora_fa        HIGH            MEASURED
medium_sl              HuggingFaceTB/SmolLM2-1.7B  llama     1.700   qlora      MEDIUM  INTERPOLATED_SCALE
large_phi                    microsoft/phi-2     phi2     2.700    lora         LOW       UNSEEN_FAMILY
    large                    Qwen/Qwen2.5-3B      qwen     3.000   qlora        HIGH            MEASURED
       xl                    Qwen/Qwen2.5-7B      qwen     7.600   qlora         LOW  OUT_OF_RANGE_SCALE

Measured envelope: 0.5-3B, families ['llama', 'qwen'], 14 directly measured
(method, scale) cells.
Scope tally: 25 OUT_OF_RANGE_SCALE, 15 INTERPOLATED_SCALE, 14 MEASURED,
             10 UNSEEN_FAMILY, 6 UNSEEN_METHOD_SCALE`
    },
    json: {
      cmd: [
        ['$ ', 'green-peft recommend', ' --vram 16 --accuracy 0.90 \\'],
        ['', '', '    --profile balanced --json --top-k 1']
      ],
      out: `{
  "n_candidates": 70,
  "n_feasible": 33,
  "n_implausible": 12,
  "profile": "balanced",
  "weights": [0.35, 0.25, 0.25, 0.15],
  "model_status": "PRELIMINARY",
  "scope_summary": "12 MEASURED, 10 INTERPOLATED_SCALE, 5 UNSEEN_FAMILY, ...",
  "ranked": [
    {
      "backbone": "medium_sl",
      "model_id": "HuggingFaceTB/SmolLM2-1.7B",
      "method": "qlora",
      "pred_accuracy": 0.9421,
      "pred_peak_vram_gb": 5.329,
      "pred_energy_kwh": 0.00359,
      "pred_carbon_kgco2eq": 0.002334,
      "pred_wall_clock_s": 197.8,
      "gei": 0.7938,
      "on_pareto_front": true,
      "confidence": "MEDIUM",
      "scope": "INTERPOLATED_SCALE"
    }
  ]
}`
    }
  };

  function cmdPlain(key) {
    return CLI[key].cmd.map(function (line) {
      if (!line) return '';
      return (line[0] ? '' : '') + line[1] + line[2];
    }).join('\n').replace(/^\n/, '').trim();
  }

  function cmdHtml(key) {
    return CLI[key].cmd.map(function (line) {
      if (!line) return '';
      const prompt = line[0] ? '<span class="c-prompt">$</span> ' : '';
      const cmd = line[1] ? '<span class="c-cmd">' + escapeHtml(line[1]) + '</span>' : '';
      const rest = escapeHtml(line[2]).replace(/(--[a-z-]+)/g, '<span class="c-flag">$1</span>');
      return prompt + cmd + rest;
    }).join('\n');
  }

  let activeTab = 'recommend';
  let typedOnce = false;

  function showTab(key, animate) {
    activeTab = key;
    $$('#cli-tabs button').forEach(function (b) {
      b.setAttribute('aria-selected', String(b.dataset.tab === key));
    });
    $('#term-cmd').innerHTML = cmdHtml(key);
    $('#term-out-label').textContent = key === 'json' ? 'json output' : 'recorded output';

    const out = $('#term-out');
    if (animate && !reduceMotion) {
      typeInto(out, CLI[key].out);
    } else {
      out.textContent = CLI[key].out;
    }
  }

  function typeInto(node, text) {
    clearInterval(typeInto._t);
    node.textContent = '';
    const caret = document.createElement('span');
    caret.className = 'term__caret';
    node.appendChild(caret);

    // Type by chunks, not characters: 1.5 kB one glyph at a time is slow enough
    // to be irritating, and the effect reads the same.
    let i = 0;
    const step = Math.max(3, Math.ceil(text.length / 260));
    typeInto._t = setInterval(function () {
      i += step;
      caret.insertAdjacentText('beforebegin', text.slice(i - step, i));
      if (i >= text.length) {
        clearInterval(typeInto._t);
        caret.remove();
      }
    }, 14);
  }

  function wireCli() {
    showTab('recommend', false);
    $$('#cli-tabs button').forEach(function (b) {
      b.addEventListener('click', function () { showTab(b.dataset.tab, true); });
    });
    $('#copy-cmd').addEventListener('click', function () {
      copy(cmdPlain(activeTab), 'Command copied');
    });
    $('#copy-repro').addEventListener('click', function () {
      copy('git clone ' + REPO + '.git\n' +
           'cd Green-AI-Decision-Support-Frameworks\n' +
           'pip install -e green_peft_cli/green_peft_pkg\n' +
           'python reproduce.py', 'Commands copied');
    });

    // The output types in once, the first time the section is actually seen.
    if (!('IntersectionObserver' in window) || reduceMotion) { typedOnce = true; return; }
    const io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting && !typedOnce) {
          typedOnce = true;
          typeInto($('#term-out'), CLI[activeTab].out);
          io.disconnect();
        }
      });
    }, { threshold: 0.25 });
    io.observe($('#cli'));
  }

  /* ================================================== 9. research ======== */

  const DOCS = [
    ['Main README', 'readme.md'],
    ['Surrogate audit report', 'docs/AUDIT_REPORT.md'],
    ['Implementation report', 'docs/IMPLEMENTATION_REPORT.md'],
    ['Paper reproduction guide', 'docs/PAPER_REPRODUCTION_GUIDE.md'],
    ['Data methodology and feature dictionary', 'docs/DATA_METHODOLOGY.md'],
    ['Primary research notebook', 'notebooks/green-peft.ipynb'],
    ['Preprocessing and feature engineering notebook', 'notebooks/data_preprocessing_and_feature_engineering.ipynb'],
    ['Benchmark execution notebook', 'notebooks/green_peft_benchmark_execution.ipynb']
  ];

  const CODE = [
    ['Reproduction script', 'reproduce.py'],
    ['Surrogate validation (grouped CV)', 'surrogate/validate.py'],
    ['Decision engine and GEI scoring', 'green_peft_cli/green_peft_pkg/green_peft/recommender.py'],
    ['Scope and confidence module', 'green_peft_cli/green_peft_pkg/green_peft/confidence.py'],
    ['Canonical benchmark artifacts', 'results/canonical_benchmark/'],
    ['Recommendation validation log', 'results/recommendation_validation.csv'],
    ['Audited model metadata', 'models/model_metadata.json'],
    ['Hugging Face export bundle', 'results/hf_export/']
  ];

  function linkList(target, rows) {
    $(target).innerHTML = rows.map(function (r) {
      // GitHub serves directories under /tree/ and files under /blob/.
      const base = r[1].endsWith('/') ? TREE : BLOB;
      return '<a href="' + base + r[1] + '" target="_blank" rel="noopener">' +
        '<span><span class="linklist__t">' + escapeHtml(r[0]) + '</span>' +
        '<span class="linklist__p">' + escapeHtml(r[1]) + '</span></span>' +
        '<svg class="linklist__arrow" width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">' +
        '<path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
      '</a>';
    }).join('');
  }

  /* ================================================== 10. chrome ========= */

  function wireChrome() {
    $('#copy-bib').addEventListener('click', function () {
      copy($('#bibtex').textContent, 'BibTeX copied');
    });

    $('#build-stamp').textContent =
      'surrogate ' + GPEngine.data.model_status.toLowerCase() +
      ' · schema ' + GPEngine.data.schema_version +
      ' · benchmark ' + EV.manifest.generated_at.slice(0, 10);

    // Scroll progress + current section in the nav.
    const bar = $('#progress');
    const links = $$('.nav a');
    const sections = links.map(function (a) { return document.querySelector(a.getAttribute('href')); });

    let ticking = false;
    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        const max = document.documentElement.scrollHeight - window.innerHeight;
        bar.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + '%';

        let current = -1;
        sections.forEach(function (s, i) {
          if (s && s.getBoundingClientRect().top <= 140) current = i;
        });
        links.forEach(function (a, i) {
          if (i === current) a.setAttribute('aria-current', 'true');
          else a.removeAttribute('aria-current');
        });
        ticking = false;
      });
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    // Reveal-on-scroll for the editorial blocks and the pipeline stages.
    if (!('IntersectionObserver' in window)) {
      $$('.reveal, .pipeline').forEach(function (n) { n.classList.add('is-in'); });
      return;
    }
    const io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); }
      });
    }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });
    $$('.reveal, .pipeline').forEach(function (n) { io.observe(n); });
  }

  /* ================================================== boot =============== */

  function boot() {
    if (!window.GREENPEFT_CANDIDATES || !window.GREENPEFT_EVIDENCE) {
      document.body.insertAdjacentHTML('afterbegin',
        '<p style="padding:20px;font-family:monospace;color:#d7a13f">' +
        'Data layer missing — run <b>python analysis/build_website_data.py</b> to generate ' +
        'website/data/candidates.js and website/data/evidence.js.</p>');
      return;
    }

    buildChipGroups();
    wireControls();
    runLab();

    buildCascade();
    buildShift();
    buildPipeline();

    buildBenchStats();
    buildEvidenceCharts();
    buildFeasibilityMatrix();
    buildMeasuredTable();
    buildFigures();

    wireGei();

    buildCvTable();
    buildBacktestTable();
    buildEnvelopeStats();

    wireCli();
    linkList('#links-docs', DOCS);
    linkList('#links-code', CODE);
    wireChrome();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
