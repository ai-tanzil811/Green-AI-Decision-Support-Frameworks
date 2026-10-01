/* ===========================================================================
   Hand-rolled SVG chart primitives.

   No chart library: every mark here is a few elements, the data is small, and
   a dependency would cost more bytes than the drawing code. Charts scale by
   viewBox, so they stay sharp and keep a stable aspect ratio on any viewport.
   ========================================================================= */

const GPCharts = (function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, text) {
    const node = document.createElementNS(NS, tag);
    if (attrs) {
      for (const k in attrs) {
        if (attrs[k] === null || attrs[k] === undefined) continue;
        node.setAttribute(k, attrs[k]);
      }
    }
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function svg(w, h, cls) {
    const s = el('svg', {
      viewBox: '0 0 ' + w + ' ' + h,
      class: cls || 'chart',
      role: 'img',
      focusable: 'false'
    });
    return s;
  }

  function scale(d0, d1, r0, r1) {
    const span = (d1 - d0) || 1;
    return function (v) { return r0 + ((v - d0) / span) * (r1 - r0); };
  }

  // Round tick steps (1, 2, 2.5, 5, 10 × 10ⁿ) so axis labels read cleanly.
  function ticks(min, max, count) {
    const span = (max - min) || 1;
    const raw = span / (count || 5);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    let step;
    if (norm <= 1) step = 1; else if (norm <= 2) step = 2;
    else if (norm <= 2.5) step = 2.5; else if (norm <= 5) step = 5; else step = 10;
    step *= mag;
    const out = [];
    for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
      out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    }
    return out;
  }

  function pad(min, max, frac) {
    const span = (max - min) || Math.abs(max) || 1;
    const p = span * (frac === undefined ? 0.08 : frac);
    return [min - p, max + p];
  }

  /* ---- marker shapes, matching the repository's matplotlib figures ------- */

  const SHAPES = {
    full_ft: 'circle',
    lisa: 'square',
    lora: 'triangle',
    lora_fa: 'diamond',
    qlora: 'triangle-down'
  };

  function marker(shape, x, y, r, attrs) {
    const a = Object.assign({}, attrs);
    switch (shape) {
      case 'square':
        return el('rect', Object.assign(a, { x: x - r, y: y - r, width: r * 2, height: r * 2, rx: 0.8 }));
      case 'triangle':
        return el('path', Object.assign(a, {
          d: 'M' + x + ' ' + (y - r * 1.15) + 'L' + (x + r * 1.1) + ' ' + (y + r * 0.8) +
             'L' + (x - r * 1.1) + ' ' + (y + r * 0.8) + 'Z'
        }));
      case 'triangle-down':
        return el('path', Object.assign(a, {
          d: 'M' + x + ' ' + (y + r * 1.15) + 'L' + (x + r * 1.1) + ' ' + (y - r * 0.8) +
             'L' + (x - r * 1.1) + ' ' + (y - r * 0.8) + 'Z'
        }));
      case 'diamond':
        return el('path', Object.assign(a, {
          d: 'M' + x + ' ' + (y - r * 1.25) + 'L' + (x + r * 1.15) + ' ' + y +
             'L' + x + ' ' + (y + r * 1.25) + 'L' + (x - r * 1.15) + ' ' + y + 'Z'
        }));
      default:
        return el('circle', Object.assign(a, { cx: x, cy: y, r: r }));
    }
  }

  /* ---- generic scatter --------------------------------------------------- */

  /**
   * opts: { points:[{x,y,label,shape,filled,color,size,title}],
   *         xLabel, yLabel, xFormat, yFormat, width, height,
   *         rules:[{axis:'x'|'y', value, text, cls}], labelled:Boolean }
   */
  function scatter(opts) {
    const W = opts.width || 520;
    const H = opts.height || 330;
    const m = Object.assign({ t: 16, r: 16, b: 40, l: 50 }, opts.margin);
    const s = svg(W, H);

    const xs = opts.points.map(function (p) { return p.x; });
    const ys = opts.points.map(function (p) { return p.y; });
    (opts.rules || []).forEach(function (r) {
      if (r.axis === 'x') xs.push(r.value); else ys.push(r.value);
    });

    let xd = pad(Math.min.apply(null, xs), Math.max.apply(null, xs), opts.xPad);
    let yd = pad(Math.min.apply(null, ys), Math.max.apply(null, ys), opts.yPad);
    if (opts.xDomain) xd = opts.xDomain;
    if (opts.yDomain) yd = opts.yDomain;

    const X = scale(xd[0], xd[1], m.l, W - m.r);
    const Y = scale(yd[0], yd[1], H - m.b, m.t);

    const g = el('g');

    // grid + ticks
    ticks(yd[0], yd[1], 4).forEach(function (t) {
      if (t < yd[0] || t > yd[1]) return;
      g.appendChild(el('line', { class: 'ax-grid', x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
      g.appendChild(el('text', {
        class: 'ax-tick', x: m.l - 7, y: Y(t) + 3, 'text-anchor': 'end'
      }, (opts.yFormat || String)(t)));
    });
    ticks(xd[0], xd[1], 4).forEach(function (t) {
      if (t < xd[0] || t > xd[1]) return;
      g.appendChild(el('line', { class: 'ax-grid', y1: m.t, y2: H - m.b, x1: X(t), x2: X(t) }));
      g.appendChild(el('text', {
        class: 'ax-tick', x: X(t), y: H - m.b + 15, 'text-anchor': 'middle'
      }, (opts.xFormat || String)(t)));
    });

    g.appendChild(el('line', { class: 'ax-line', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    g.appendChild(el('line', { class: 'ax-line', x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }));

    if (opts.xLabel) {
      g.appendChild(el('text', {
        class: 'ax-label', x: W - m.r, y: H - 6, 'text-anchor': 'end'
      }, opts.xLabel));
    }
    if (opts.yLabel) {
      g.appendChild(el('text', {
        class: 'ax-label', x: -m.t, y: 11, 'text-anchor': 'end',
        transform: 'rotate(-90)'
      }, opts.yLabel));
    }

    // constraint rules
    (opts.rules || []).forEach(function (r) {
      if (r.axis === 'x') {
        g.appendChild(el('line', { class: r.cls || 'budget-line', x1: X(r.value), x2: X(r.value), y1: m.t, y2: H - m.b }));
        if (r.text) {
          g.appendChild(el('text', {
            class: 'budget-text', x: X(r.value) - 5, y: m.t + 10, 'text-anchor': 'end'
          }, r.text));
        }
      } else {
        g.appendChild(el('line', { class: r.cls || 'budget-line', y1: Y(r.value), y2: Y(r.value), x1: m.l, x2: W - m.r }));
        if (r.text) {
          g.appendChild(el('text', {
            class: 'budget-text', x: W - m.r, y: Y(r.value) - 5, 'text-anchor': 'end'
          }, r.text));
        }
      }
    });

    // points
    opts.points.forEach(function (p) {
      const r = p.size || 4.2;
      const attrs = p.filled === false
        ? { fill: 'none', stroke: p.color, 'stroke-width': 1.4, class: 'pt' }
        : { fill: p.color, class: 'pt', stroke: p.ring || 'none', 'stroke-width': p.ring ? 1.6 : 0 };
      const node = marker(p.shape || 'circle', X(p.x), Y(p.y), r, attrs);
      if (p.title) node.appendChild(el('title', null, p.title));
      g.appendChild(node);
      if (p.label) {
        const dx = p.labelSide === 'left' ? -(r + 4) : (r + 4);
        g.appendChild(el('text', {
          class: 'pt-label', x: X(p.x) + dx, y: Y(p.y) + 3,
          'text-anchor': p.labelSide === 'left' ? 'end' : 'start'
        }, p.label));
      }
    });

    s.appendChild(g);
    return s;
  }

  /* ---- hero constraint cascade ------------------------------------------ */

  /**
   * A live catalogue matrix: one cell per (backbone, method) candidate, showing
   * predicted peak VRAM and changing state as each gate is applied. Stages are
   * driven from outside via the returned setStage().
   */
  function cascadeMatrix(opts) {
    const methods = opts.methods;
    const backbones = opts.backbones;
    const byKey = opts.byKey;              // "backbone|method" -> candidate
    const state = opts.state;              // "backbone|method" -> 'pass'|'vram'|'acc'|'bad'|'other'

    const LEFT = 108, TOP = 26, CW = 56, CH = 18, GX = 4, GY = 3.5;
    const W = LEFT + methods.length * CW + (methods.length - 1) * GX + 8;
    const H = TOP + backbones.length * CH + (backbones.length - 1) * GY + 8;

    const s = svg(W, H, 'cascade__svg');
    s.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // column headers
    methods.forEach(function (m, i) {
      const x = LEFT + i * (CW + GX) + CW / 2;
      s.appendChild(el('text', {
        class: 'cgate-label', x: x, y: TOP - 10, 'text-anchor': 'middle'
      }, m.replace('_', '-')));
    });

    backbones.forEach(function (b, r) {
      const y = TOP + r * (CH + GY);

      s.appendChild(el('text', {
        class: 'cgate-label', x: LEFT - 32, y: y + CH / 2 + 3, 'text-anchor': 'end'
      }, b.key.replace(/_/g, '-')));
      s.appendChild(el('text', {
        class: 'cgate-label', x: LEFT - 8, y: y + CH / 2 + 3, 'text-anchor': 'end',
        style: 'fill:#4c6358'
      }, b.params_b + 'B'));

      methods.forEach(function (m, c) {
        const key = b.key + '|' + m;
        const cand = byKey[key];
        const x = LEFT + c * (CW + GX);
        const st = state[key] || 'other';

        const cell = el('g', { class: 'ccell', 'data-key': key, opacity: 0 });
        cell.appendChild(el('rect', {
          x: x, y: y, width: CW, height: CH, rx: 1.5,
          class: 'ccell-bg ccell-bg--' + st
        }));
        cell.appendChild(el('text', {
          x: x + CW / 2, y: y + CH / 2 + 3, 'text-anchor': 'middle',
          class: 'ccell-tx ccell-tx--' + st
        }, cand && !cand.implausible ? cand.pred_peak_vram_gb.toFixed(1) : '––'));

        if (cand) {
          cell.appendChild(el('title', null,
            GPEngine.methodLabel(m) + ' on ' + b.key + ' (' + b.model_id + ')\n' +
            'predicted peak VRAM ' + (cand.implausible ? 'implausible' : cand.pred_peak_vram_gb.toFixed(2) + ' GB') +
            '\npredicted accuracy ' + cand.pred_accuracy.toFixed(4) +
            '\nscope ' + cand.scope));
        }
        s.appendChild(cell);
      });
    });

    return { node: s, width: W, height: H };
  }

  /* ---- pipeline diagram -------------------------------------------------- */

  function pipeline(stages) {
    const BW = 196, GAPX = 26;
    const W = stages.length * BW + (stages.length - 1) * GAPX;
    const H = 268;
    const s = svg(W, H, 'pipeline__svg');
    s.setAttribute('preserveAspectRatio', 'xMinYMin meet');

    stages.forEach(function (st, i) {
      const x = i * (BW + GAPX);
      const g = el('g', { class: 'pstage' });

      // inputs above the box
      g.appendChild(el('text', { class: 'pio-head', x: x, y: 10 }, 'INPUT'));
      st.in.forEach(function (t, k) {
        g.appendChild(el('text', { class: 'pio', x: x, y: 24 + k * 12 }, '· ' + t));
      });

      const boxY = 24 + Math.max(st.in.length, 1) * 12 + 10;
      const boxH = 62;
      g.appendChild(el('rect', { class: 'pbox', x: x, y: boxY, width: BW, height: boxH, rx: 3 }));
      g.appendChild(el('rect', { x: x, y: boxY, width: 2.5, height: boxH, fill: '#4fbdaf' }));
      g.appendChild(el('text', { class: 'pnum', x: x + 14, y: boxY + 20 }, 'STAGE 0' + (i + 1)));

      wrapText(g, st.title, x + 14, boxY + 38, BW - 24, 13.5, 'ptitle');

      // outputs below
      g.appendChild(el('text', { class: 'pio-head', x: x, y: boxY + boxH + 20 }, 'OUTPUT'));
      st.out.forEach(function (t, k) {
        g.appendChild(el('text', { class: 'pio', x: x, y: boxY + boxH + 34 + k * 12 }, '· ' + t));
      });

      // flow connector to the next stage
      if (i < stages.length - 1) {
        const cy = boxY + boxH / 2;
        g.appendChild(el('path', {
          class: 'pflow pflow--dash',
          d: 'M' + (x + BW) + ' ' + cy + 'H' + (x + BW + GAPX - 7)
        }));
        g.appendChild(el('path', {
          class: 'parrow',
          d: 'M' + (x + BW + GAPX - 8) + ' ' + (cy - 3.5) + 'l6 3.5-6 3.5z'
        }));
      }
      s.appendChild(g);
    });

    return { node: s, width: W, height: H };
  }

  // Minimal greedy wrap — SVG has no text flow, and the strings here are short
  // and known, so an approximate per-character width is accurate enough.
  function wrapText(parent, text, x, y, maxWidth, lineHeight, cls) {
    const words = text.split(' ');
    const perChar = 6.6;
    let line = [], lines = [];
    words.forEach(function (w) {
      const trial = line.concat([w]);
      if (trial.join(' ').length * perChar > maxWidth && line.length) {
        lines.push(line.join(' '));
        line = [w];
      } else {
        line = trial;
      }
    });
    if (line.length) lines.push(line.join(' '));
    lines.forEach(function (l, i) {
      parent.appendChild(el('text', { class: cls, x: x, y: y + i * lineHeight }, l));
    });
    return lines.length;
  }

  return {
    el: el, svg: svg, scale: scale, ticks: ticks, pad: pad,
    marker: marker, SHAPES: SHAPES,
    scatter: scatter, cascadeMatrix: cascadeMatrix, pipeline: pipeline
  };
})();
