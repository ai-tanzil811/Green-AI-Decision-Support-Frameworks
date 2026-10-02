/* ============================================================================
   GreenPEFT — Hand-Crafted SVG Charts (Monochrome Dark & Light Adaptive)
   ============================================================================ */

window.GPCharts = (function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';

  function isDarkMode() {
    return document.documentElement.getAttribute('data-theme') !== 'light';
  }

  function getThemeColors() {
    const dark = isDarkMode();
    return {
      ink: dark ? '#FFFFFF' : '#000000',
      inkMuted: dark ? '#A3A3A3' : '#4D4D4D',
      inkDim: dark ? '#767676' : '#767676',
      grid: dark ? '#222222' : '#EDEDED',
      border: dark ? '#262626' : '#EDEDED',
      pointFeasible: dark ? '#FFFFFF' : '#000000',
      pointFiltered: dark ? '#2E2E2E' : '#D4D4D4',
      pointFilteredStroke: dark ? '#444444' : '#A3A3A3',
      pointStroke: dark ? '#000000' : '#FFFFFF',
      vramLine: dark ? '#FFFFFF' : '#000000',
      vramHandle: dark ? '#FFFFFF' : '#000000',
      vramGrip: dark ? '#000000' : '#FFFFFF'
    };
  }

  function createSvgEl(tag, attrs, text) {
    const el = document.createElementNS(NS, tag);
    if (attrs) {
      for (const k in attrs) {
        if (attrs[k] !== null && attrs[k] !== undefined) {
          el.setAttribute(k, attrs[k]);
        }
      }
    }
    if (text !== undefined && text !== null) {
      el.textContent = text;
    }
    return el;
  }

  /* --------------------------------------------------------------------------
     1. Hero Interactive Scatter Plot (with Draggable VRAM Line)
     ----------------------------------------------------------------------- */
  let heroChartInstance = null;

  function initHeroScatter(containerEl, onVramChange) {
    if (!containerEl) return;
    containerEl.innerHTML = '';

    const colors = getThemeColors();
    const width = 520;
    const height = 340;
    const margin = { top: 24, right: 32, bottom: 44, left: 54 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    // Match the recommender's full VRAM control range.
    const xMin = 0, xMax = 48;
    const yMin = 0.86, yMax = 0.97;

    function scaleX(val) {
      return margin.left + ((val - xMin) / (xMax - xMin)) * innerW;
    }
    function unscaleX(px) {
      const clamped = Math.max(margin.left, Math.min(margin.left + innerW, px));
      return xMin + ((clamped - margin.left) / innerW) * (xMax - xMin);
    }
    function scaleY(val) {
      return margin.top + innerH - ((val - yMin) / (yMax - yMin)) * innerH;
    }

    const pointsData = [
      { id: 'qlora_tiny',   method: 'QLoRA',   backbone: '0.5B Tiny',   vram: 2.76,  acc: 0.912, gei: 0.687, labelSide: 'top' },
      { id: 'lora_fa_tiny', method: 'LoRA-FA', backbone: '0.5B Tiny',   vram: 4.30,  acc: 0.916, gei: 0.759, labelSide: 'bottom' },
      { id: 'lora_tiny',    method: 'LoRA',    backbone: '0.5B Tiny',   vram: 4.34,  acc: 0.917, gei: 0.770, labelSide: 'top' },
      { id: 'lisa_tiny',    method: 'LISA',    backbone: '0.5B Tiny',   vram: 5.11,  acc: 0.887, gei: 0.601, labelSide: 'bottom' },
      { id: 'qlora_small',  method: 'QLoRA',   backbone: '1.1B Small',  vram: 3.96,  acc: 0.938, gei: 0.789, labelSide: 'top' },
      { id: 'qlora_med',    method: 'QLoRA',   backbone: '1.5B Med',    vram: 6.42,  acc: 0.943, gei: 0.711, labelSide: 'top' },
      { id: 'qlora_large',  method: 'QLoRA',   backbone: '3.0B Large',  vram: 7.14,  acc: 0.948, gei: 0.512, labelSide: 'top' },
      { id: 'lora_fa_sm',   method: 'LoRA-FA', backbone: '1.1B Small',  vram: 8.38,  acc: 0.925, gei: 0.712, labelSide: 'bottom' },
      { id: 'lora_small',   method: 'LoRA',    backbone: '1.1B Small',  vram: 9.43,  acc: 0.944, gei: 0.792, labelSide: 'top' },
      { id: 'lisa_small',   method: 'LISA',    backbone: '1.1B Small',  vram: 10.88, acc: 0.920, gei: 0.606, labelSide: 'bottom' },
      { id: 'full_ft_tiny', method: 'Full-FT', backbone: '0.5B Tiny',   vram: 11.13, acc: 0.905, gei: 0.490, labelSide: 'bottom' },
      { id: 'lora_med',     method: 'LoRA',    backbone: '1.5B Med',    vram: 11.97, acc: 0.948, gei: 0.730, labelSide: 'top' },
      { id: 'lisa_med',     method: 'LISA',    backbone: '1.5B Med',    vram: 15.21, acc: 0.938, gei: 0.847, labelSide: 'top' }
    ];

    const svg = createSvgEl('svg', {
      viewBox: `0 0 ${width} ${height}`,
      class: 'hero-scatter-svg',
      role: 'region',
      'aria-label': 'Interactive Pareto scatter plot: Accuracy vs VRAM with draggable VRAM line'
    });

    // Grid lines & Axis labels
    const gridG = createSvgEl('g', { class: 'grid-lines' });
    const yTicks = [0.86, 0.88, 0.90, 0.92, 0.94, 0.96];
    yTicks.forEach(t => {
      const y = scaleY(t);
      gridG.appendChild(createSvgEl('line', {
        x1: margin.left, x2: width - margin.right, y1: y, y2: y,
        stroke: colors.grid, 'stroke-width': '1'
      }));
      gridG.appendChild(createSvgEl('text', {
        x: margin.left - 8, y: y + 4,
        'text-anchor': 'end',
        'font-family': 'JetBrains Mono',
        'font-size': '11',
        fill: colors.inkDim
      }, t.toFixed(2)));
    });

    const xTicks = [0, 8, 16, 24, 32, 40, 48];
    xTicks.forEach(t => {
      const x = scaleX(t);
      gridG.appendChild(createSvgEl('line', {
        x1: x, x2: x, y1: margin.top, y2: height - margin.bottom,
        stroke: colors.grid, 'stroke-width': '1'
      }));
      gridG.appendChild(createSvgEl('text', {
        x: x, y: height - margin.bottom + 18,
        'text-anchor': 'middle',
        'font-family': 'JetBrains Mono',
        'font-size': '11',
        fill: colors.inkDim
      }, t + 'G'));
    });

    // Axis titles
    gridG.appendChild(createSvgEl('text', {
      x: width - margin.right, y: height - 8,
      'text-anchor': 'end',
      'font-family': 'JetBrains Mono',
      'font-size': '11',
      fill: colors.ink,
      'font-weight': '500'
    }, 'Peak VRAM (GB) →'));

    gridG.appendChild(createSvgEl('text', {
      x: -margin.top, y: 16,
      'text-anchor': 'end',
      transform: 'rotate(-90)',
      'font-family': 'JetBrains Mono',
      'font-size': '11',
      fill: colors.ink,
      'font-weight': '500'
    }, 'Accuracy →'));

    // Base axes lines
    gridG.appendChild(createSvgEl('line', {
      x1: margin.left, x2: width - margin.right,
      y1: height - margin.bottom, y2: height - margin.bottom,
      stroke: colors.ink, 'stroke-width': '1'
    }));
    gridG.appendChild(createSvgEl('line', {
      x1: margin.left, x2: margin.left,
      y1: margin.top, y2: height - margin.bottom,
      stroke: colors.ink, 'stroke-width': '1'
    }));

    svg.appendChild(gridG);

    // Points group
    const pointsG = createSvgEl('g', { class: 'scatter-points' });
    const pointElements = [];

    pointsData.forEach(p => {
      const cx = scaleX(p.vram);
      const cy = scaleY(p.acc);

      const pg = createSvgEl('g', {
        class: 'scatter-point-group',
        'data-vram': p.vram,
        'data-acc': p.acc,
        'data-gei': p.gei,
        'data-name': `${p.method} (${p.backbone})`,
        tabindex: '0',
        style: 'cursor: pointer; transition: opacity 180ms ease;'
      });

      // Dot
      const circle = createSvgEl('circle', {
        cx: cx, cy: cy, r: 5,
        fill: colors.pointFeasible,
        stroke: colors.pointStroke,
        'stroke-width': '1.5'
      });
      pg.appendChild(circle);

      // Method label
      const dy = p.labelSide === 'bottom' ? 14 : -9;
      const lbl = createSvgEl('text', {
        class: 'scatter-point-label',
        x: cx, y: cy + dy,
        'text-anchor': 'middle',
        'font-family': 'JetBrains Mono',
        'font-size': '9.5',
        'font-weight': '500',
        fill: colors.ink
      }, p.method);
      pg.appendChild(lbl);

      pg.addEventListener('mouseenter', (e) => showTooltip(e, p));
      pg.addEventListener('focus', (e) => showTooltip(e, p));
      pg.addEventListener('mouseleave', hideTooltip);
      pg.addEventListener('blur', hideTooltip);

      pointsG.appendChild(pg);
      pointElements.push({ data: p, group: pg, circle: circle, label: lbl });
    });

    svg.appendChild(pointsG);

    // Draggable VRAM Line & Handle Group
    let currentVram = 16.0;
    const vramLineG = createSvgEl('g', {
      class: 'vram-drag-group',
      style: 'cursor: ew-resize;'
    });

    const vramLine = createSvgEl('line', {
      x1: scaleX(currentVram), x2: scaleX(currentVram),
      y1: margin.top, y2: height - margin.bottom,
      stroke: colors.vramLine,
      'stroke-width': '1.5',
      'stroke-dasharray': '3,3'
    });
    vramLineG.appendChild(vramLine);

    // Draggable Handle
    const handleW = 20, handleH = 20;
    const handleRect = createSvgEl('rect', {
      x: scaleX(currentVram) - handleW / 2,
      y: margin.top - 6,
      width: handleW, height: handleH,
      fill: colors.vramHandle,
      rx: '2',
      tabindex: '0',
      role: 'slider',
      'aria-valuemin': '4',
      'aria-valuemax': '48',
      'aria-valuenow': currentVram.toFixed(1),
      'aria-label': 'VRAM constraint boundary in GB'
    });
    vramLineG.appendChild(handleRect);

    // Grip lines on handle
    const grip1 = createSvgEl('line', {
      x1: scaleX(currentVram) - 2, x2: scaleX(currentVram) - 2,
      y1: margin.top - 1, y2: margin.top + 9,
      stroke: colors.vramGrip, 'stroke-width': '1'
    });
    const grip2 = createSvgEl('line', {
      x1: scaleX(currentVram) + 2, x2: scaleX(currentVram) + 2,
      y1: margin.top - 1, y2: margin.top + 9,
      stroke: colors.vramGrip, 'stroke-width': '1'
    });
    vramLineG.appendChild(grip1);
    vramLineG.appendChild(grip2);

    // Tag label above line
    const tagText = createSvgEl('text', {
      x: scaleX(currentVram), y: margin.top - 10,
      'text-anchor': 'middle',
      'font-family': 'JetBrains Mono',
      'font-size': '10',
      'font-weight': '600',
      fill: colors.ink
    }, `${currentVram.toFixed(1)} GB`);
    vramLineG.appendChild(tagText);

    svg.appendChild(vramLineG);
    containerEl.appendChild(svg);

    // Update filter state based on VRAM threshold
    function updateVramThreshold(vramVal, triggerCallback) {
      currentVram = Math.max(2, Math.min(48, vramVal));
      const px = scaleX(currentVram);
      const c = getThemeColors();

      vramLine.setAttribute('x1', px);
      vramLine.setAttribute('x2', px);
      handleRect.setAttribute('x', px - handleW / 2);
      handleRect.setAttribute('aria-valuenow', currentVram.toFixed(1));
      grip1.setAttribute('x1', px - 2);
      grip1.setAttribute('x2', px - 2);
      grip2.setAttribute('x1', px + 2);
      grip2.setAttribute('x2', px + 2);
      tagText.setAttribute('x', px);
      tagText.textContent = `${currentVram.toFixed(1)} GB`;

      let feasibleCount = 0;
      pointElements.forEach(pt => {
        const isVisible = pt.group.getAttribute('data-method-visible') !== 'false';
        const isFeasible = isVisible && pt.data.vram <= currentVram;
        if (isFeasible) {
          feasibleCount++;
          pt.group.style.opacity = '1';
          pt.circle.setAttribute('fill', c.pointFeasible);
          pt.label.setAttribute('fill', c.ink);
        } else {
          pt.group.style.opacity = '0.25';
          pt.circle.setAttribute('fill', c.pointFiltered);
          pt.label.setAttribute('fill', c.inkDim);
        }
      });

      const readout = document.getElementById('hero-vram-readout');
      if (readout) {
        readout.textContent = `${currentVram.toFixed(1)} GB (${feasibleCount} feasible)`;
      }

      if (triggerCallback && typeof onVramChange === 'function') {
        onVramChange(currentVram);
      }
    }

    function setMethodVisibility(method, visible) {
      pointElements.forEach(pt => {
        if (pt.data.method !== method) return;
        pt.group.setAttribute('data-method-visible', String(visible));
        pt.group.style.display = visible ? '' : 'none';
      });
      updateVramThreshold(currentVram, false);
    }

    pointElements.forEach(pt => pt.group.setAttribute('data-method-visible', 'true'));
    document.querySelectorAll('#hero-method-legend .chart-legend-btn').forEach(button => {
      button.onclick = () => {
        const active = button.getAttribute('aria-pressed') === 'true';
        button.setAttribute('aria-pressed', String(!active));
        button.classList.toggle('is-active', !active);
        setMethodVisibility(button.getAttribute('data-method'), !active);
      };
      if (button.getAttribute('aria-pressed') !== 'true') {
        setMethodVisibility(button.getAttribute('data-method'), false);
      }
    });

    // Drag event handling (Mouse + Touch)
    let isDragging = false;

    function getSvgCoordinates(evt) {
      const rect = svg.getBoundingClientRect();
      const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
      const svgX = ((clientX - rect.left) / rect.width) * width;
      return unscaleX(svgX);
    }

    function onPointerDown(e) {
      isDragging = true;
      e.preventDefault();
      updateVramThreshold(getSvgCoordinates(e), true);
    }

    function onPointerMove(e) {
      if (!isDragging) return;
      e.preventDefault();
      updateVramThreshold(getSvgCoordinates(e), true);
    }

    function onPointerUp() {
      isDragging = false;
    }

    vramLineG.addEventListener('mousedown', onPointerDown);
    vramLineG.addEventListener('touchstart', onPointerDown, { passive: false });
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('touchmove', onPointerMove, { passive: false });
    window.addEventListener('mouseup', onPointerUp);
    window.addEventListener('touchend', onPointerUp);

    // Keyboard support for handle
    handleRect.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
        e.preventDefault();
        updateVramThreshold(currentVram - 0.5, true);
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
        e.preventDefault();
        updateVramThreshold(currentVram + 0.5, true);
      }
    });

    // Initial run
    updateVramThreshold(16.0, false);

    heroChartInstance = {
      setVram: function (val) {
        updateVramThreshold(val, false);
      },
      rebuild: function () {
        initHeroScatter(containerEl, onVramChange);
      },
      setMethodVisibility: setMethodVisibility
    };

    return heroChartInstance;
  }

  /* --------------------------------------------------------------------------
     2. Benchmark Evidence Scatter (60-run Suite)
     ----------------------------------------------------------------------- */
  function initEvidenceScatter(containerEl) {
    if (!containerEl) return;
    containerEl.innerHTML = '';

    const colors = getThemeColors();
    const width = 520;
    const height = 340;
    const margin = { top: 24, right: 28, bottom: 44, left: 54 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const xMin = 0, xMax = 18;
    const yMin = 0.87, yMax = 0.96;

    function scaleX(val) {
      return margin.left + ((val - xMin) / (xMax - xMin)) * innerW;
    }
    function scaleY(val) {
      return margin.top + innerH - ((val - yMin) / (yMax - yMin)) * innerH;
    }

    const svg = createSvgEl('svg', {
      viewBox: `0 0 ${width} ${height}`,
      class: 'evidence-svg',
      role: 'region',
      'aria-label': '60-run empirical benchmark: Accuracy vs Peak VRAM'
    });

    // Grid lines
    [0.88, 0.90, 0.92, 0.94, 0.96].forEach(t => {
      const y = scaleY(t);
      svg.appendChild(createSvgEl('line', {
        x1: margin.left, x2: width - margin.right, y1: y, y2: y,
        stroke: colors.grid, 'stroke-width': '1'
      }));
      svg.appendChild(createSvgEl('text', {
        x: margin.left - 8, y: y + 4,
        'text-anchor': 'end',
        'font-family': 'JetBrains Mono',
        'font-size': '11',
        fill: colors.inkDim
      }, t.toFixed(2)));
    });

    [0, 4, 8, 12, 16].forEach(t => {
      const x = scaleX(t);
      svg.appendChild(createSvgEl('line', {
        x1: x, x2: x, y1: margin.top, y2: height - margin.bottom,
        stroke: colors.grid, 'stroke-width': '1'
      }));
      svg.appendChild(createSvgEl('text', {
        x: x, y: height - margin.bottom + 18,
        'text-anchor': 'middle',
        'font-family': 'JetBrains Mono',
        'font-size': '11',
        fill: colors.inkDim
      }, t + ' GB'));
    });

    // Axes
    svg.appendChild(createSvgEl('line', {
      x1: margin.left, x2: width - margin.right,
      y1: height - margin.bottom, y2: height - margin.bottom,
      stroke: colors.ink, 'stroke-width': '1'
    }));
    svg.appendChild(createSvgEl('line', {
      x1: margin.left, x2: margin.left,
      y1: margin.top, y2: height - margin.bottom,
      stroke: colors.ink, 'stroke-width': '1'
    }));

    // Axis Labels
    svg.appendChild(createSvgEl('text', {
      x: width - margin.right, y: height - 8,
      'text-anchor': 'end',
      'font-family': 'JetBrains Mono',
      'font-size': '11',
      fill: colors.ink,
      'font-weight': '500'
    }, 'Peak VRAM (GB) →'));

    svg.appendChild(createSvgEl('text', {
      x: -margin.top, y: 16,
      'text-anchor': 'end',
      transform: 'rotate(-90)',
      'font-family': 'JetBrains Mono',
      'font-size': '11',
      fill: colors.ink,
      'font-weight': '500'
    }, 'Accuracy (SST-2) →'));

    const measuredPoints = (window.GREENPEFT_EVIDENCE && window.GREENPEFT_EVIDENCE.measured) || [
      { method: 'lora', backbone: 'small', accuracy: 0.9438, peak_gpu_memory_gb: 9.43, gei_balanced: 0.792 },
      { method: 'qlora', backbone: 'small', accuracy: 0.9381, peak_gpu_memory_gb: 3.96, gei_balanced: 0.789 },
      { method: 'lora', backbone: 'tiny', accuracy: 0.9174, peak_gpu_memory_gb: 4.34, gei_balanced: 0.770 },
      { method: 'lora_fa', backbone: 'tiny', accuracy: 0.9155, peak_gpu_memory_gb: 4.30, gei_balanced: 0.759 },
      { method: 'lora', backbone: 'medium', accuracy: 0.9478, peak_gpu_memory_gb: 11.97, gei_balanced: 0.730 },
      { method: 'lora_fa', backbone: 'small', accuracy: 0.9251, peak_gpu_memory_gb: 8.38, gei_balanced: 0.712 },
      { method: 'qlora', backbone: 'medium', accuracy: 0.9434, peak_gpu_memory_gb: 6.42, gei_balanced: 0.711 },
      { method: 'lora_fa', backbone: 'medium', accuracy: 0.9430, peak_gpu_memory_gb: 12.20, gei_balanced: 0.702 },
      { method: 'qlora', backbone: 'tiny', accuracy: 0.9117, peak_gpu_memory_gb: 2.76, gei_balanced: 0.687 },
      { method: 'lisa', backbone: 'small', accuracy: 0.9201, peak_gpu_memory_gb: 10.88, gei_balanced: 0.606 },
      { method: 'lisa', backbone: 'tiny', accuracy: 0.8868, peak_gpu_memory_gb: 5.11, gei_balanced: 0.601 },
      { method: 'lisa', backbone: 'medium', accuracy: 0.9381, peak_gpu_memory_gb: 15.21, gei_balanced: 0.585 },
      { method: 'qlora', backbone: 'large', accuracy: 0.9484, peak_gpu_memory_gb: 7.14, gei_balanced: 0.512 },
      { method: 'full_ft', backbone: 'tiny', accuracy: 0.9052, peak_gpu_memory_gb: 11.13, gei_balanced: 0.490 }
    ];

    measuredPoints.forEach(pt => {
      const cx = scaleX(pt.peak_gpu_memory_gb);
      const cy = scaleY(pt.accuracy);

      const g = createSvgEl('g', {
        class: 'evidence-pt',
        tabindex: '0',
        style: 'cursor: pointer;'
      });

      if (pt.method === 'lisa') {
        g.appendChild(createSvgEl('rect', {
          x: cx - 4, y: cy - 4, width: 8, height: 8,
          fill: colors.pointFeasible, stroke: colors.pointStroke, 'stroke-width': '1.2'
        }));
      } else if (pt.method === 'lora') {
        g.appendChild(createSvgEl('path', {
          d: `M${cx},${cy - 5} L${cx + 5},${cy + 4} L${cx - 5},${cy + 4} Z`,
          fill: colors.pointFeasible, stroke: colors.pointStroke, 'stroke-width': '1.2'
        }));
      } else if (pt.method === 'qlora') {
        g.appendChild(createSvgEl('path', {
          d: `M${cx - 5},${cy - 4} L${cx + 5},${cy - 4} L${cx},${cy + 5} Z`,
          fill: colors.pointFeasible, stroke: colors.pointStroke, 'stroke-width': '1.2'
        }));
      } else if (pt.method === 'lora_fa') {
        g.appendChild(createSvgEl('path', {
          d: `M${cx},${cy - 5} L${cx + 5},${cy} L${cx},${cy + 5} L${cx - 5},${cy} Z`,
          fill: colors.pointFeasible, stroke: colors.pointStroke, 'stroke-width': '1.2'
        }));
      } else {
        g.appendChild(createSvgEl('circle', {
          cx: cx, cy: cy, r: 4.5,
          fill: colors.pointFeasible, stroke: colors.pointStroke, 'stroke-width': '1.2'
        }));
      }

      const pointInfo = {
        method: pt.method.toUpperCase().replace('_', '-'),
        backbone: pt.backbone,
        vram: pt.peak_gpu_memory_gb,
        acc: pt.accuracy,
        gei: pt.gei_balanced
      };

      g.addEventListener('mouseenter', (e) => showTooltip(e, pointInfo));
      g.addEventListener('focus', (e) => showTooltip(e, pointInfo));
      g.addEventListener('mouseleave', hideTooltip);
      g.addEventListener('blur', hideTooltip);

      svg.appendChild(g);
    });

    // 16 GB hardware limit indicator
    const x16 = scaleX(16.0);
    svg.appendChild(createSvgEl('line', {
      x1: x16, x2: x16, y1: margin.top, y2: height - margin.bottom,
      stroke: colors.inkDim, 'stroke-width': '1', 'stroke-dasharray': '2,2'
    }));
    svg.appendChild(createSvgEl('text', {
      x: x16 - 4, y: margin.top + 14,
      'text-anchor': 'end',
      'font-family': 'JetBrains Mono',
      'font-size': '10',
      fill: colors.inkDim
    }, '16 GB T4 limit'));

    containerEl.appendChild(svg);
  }

  /* --------------------------------------------------------------------------
     3. Why It Matters: Section 3 Beat 3 (Filter First vs Run Everything)
     ----------------------------------------------------------------------- */
  let filterSvgInstance = null;

  function initFilterToggleSvg(containerEl) {
    if (!containerEl) return;
    containerEl.innerHTML = '';

    const colors = getThemeColors();
    const width = 480;
    const height = 240;
    const margin = { top: 20, right: 24, bottom: 32, left: 44 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = createSvgEl('svg', {
      viewBox: `0 0 ${width} ${height}`,
      class: 'filter-dots-svg',
      role: 'img',
      'aria-label': 'Visual comparison: unguided grid sweep vs Pareto filtered shortlist'
    });

    const candidateDots = [
      { x: 30,  y: 40, isPareto: false, name: 'Config A' },
      { x: 50,  y: 85, isPareto: true,  name: 'QLoRA 1.1B' },
      { x: 75,  y: 50, isPareto: false, name: 'Config C' },
      { x: 90,  y: 70, isPareto: false, name: 'Config D' },
      { x: 120, y: 92, isPareto: true,  name: 'LoRA 1.1B' },
      { x: 140, y: 60, isPareto: false, name: 'Config F' },
      { x: 160, y: 78, isPareto: false, name: 'Config G' },
      { x: 180, y: 96, isPareto: true,  name: 'LISA 1.5B' },
      { x: 210, y: 88, isPareto: false, name: 'Config I' },
      { x: 230, y: 65, isPareto: false, name: 'Config J' },
      { x: 260, y: 75, isPareto: false, name: 'Config K' },
      { x: 290, y: 90, isPareto: false, name: 'Full-FT 1.5B (OOM)' }
    ];

    // Grid baseline
    svg.appendChild(createSvgEl('line', {
      x1: margin.left, x2: width - margin.right,
      y1: height - margin.bottom, y2: height - margin.bottom,
      stroke: colors.grid, 'stroke-width': '1'
    }));
    svg.appendChild(createSvgEl('line', {
      x1: margin.left, x2: margin.left,
      y1: margin.top, y2: height - margin.bottom,
      stroke: colors.grid, 'stroke-width': '1'
    }));

    svg.appendChild(createSvgEl('text', {
      x: width - margin.right, y: height - 10,
      'text-anchor': 'end',
      'font-family': 'JetBrains Mono',
      'font-size': '10',
      fill: colors.inkDim
    }, 'Compute cost / VRAM →'));

    svg.appendChild(createSvgEl('text', {
      x: -margin.top, y: 16,
      'text-anchor': 'end',
      transform: 'rotate(-90)',
      'font-family': 'JetBrains Mono',
      'font-size': '10',
      fill: colors.inkDim
    }, 'Accuracy →'));

    // Connecting Pareto frontier line
    const paretoPath = createSvgEl('path', {
      d: 'M 102 70 L 176 56 L 244 48',
      fill: 'none',
      stroke: colors.ink,
      'stroke-width': '1.5',
      'stroke-dasharray': '3,3',
      style: 'opacity: 0; transition: opacity 220ms ease;'
    });
    svg.appendChild(paretoPath);

    const dotElements = [];

    candidateDots.forEach(d => {
      const cx = margin.left + (d.x / 320) * innerW;
      const cy = height - margin.bottom - (d.y / 110) * innerH;

      const dot = createSvgEl('circle', {
        cx: cx, cy: cy, r: 5,
        fill: colors.pointFeasible,
        stroke: colors.pointStroke,
        'stroke-width': '1.5',
        style: 'transition: all 220ms ease;'
      });

      const label = createSvgEl('text', {
        x: cx, y: cy - 9,
        'text-anchor': 'middle',
        'font-family': 'JetBrains Mono',
        'font-size': '10',
        'font-weight': '600',
        fill: colors.ink,
        style: 'opacity: 0; transition: opacity 220ms ease;'
      }, d.name);

      svg.appendChild(dot);
      svg.appendChild(label);
      dotElements.push({ data: d, dot: dot, label: label, cx: cx, cy: cy });
    });

    containerEl.appendChild(svg);

    function setState(mode) {
      const c = getThemeColors();
      if (mode === 'greenpeft') {
        paretoPath.style.opacity = '1';
        dotElements.forEach(item => {
          if (item.data.isPareto) {
            item.dot.setAttribute('fill', c.pointFeasible);
            item.dot.setAttribute('r', '6');
            item.dot.style.opacity = '1';
            item.label.style.opacity = '1';
          } else {
            item.dot.setAttribute('fill', c.pointFiltered);
            item.dot.setAttribute('r', '4');
            item.dot.style.opacity = '0.2';
            item.label.style.opacity = '0';
          }
        });
      } else {
        paretoPath.style.opacity = '0';
        dotElements.forEach(item => {
          item.dot.setAttribute('fill', c.pointFeasible);
          item.dot.setAttribute('r', '5');
          item.dot.style.opacity = '0.7';
          item.label.style.opacity = '0';
        });
      }
    }

    setState('run_all');

    filterSvgInstance = {
      setState: setState,
      rebuild: function () {
        initFilterToggleSvg(containerEl);
      }
    };

    return filterSvgInstance;
  }

  /* --------------------------------------------------------------------------
     4. Tooltip System (Single Instance, Accessible)
     ----------------------------------------------------------------------- */
  let tooltipEl = null;

  function ensureTooltip() {
    if (!tooltipEl) {
      tooltipEl = document.createElement('div');
      tooltipEl.className = 'gp-tooltip';
      tooltipEl.setAttribute('role', 'tooltip');
      document.body.appendChild(tooltipEl);
    }
  }

  function showTooltip(evt, data) {
    ensureTooltip();
    let content = '';
    if (data.name) {
      content += `<div style="font-weight:600;margin-bottom:4px;">${data.name}</div>`;
    } else if (data.method) {
      content += `<div style="font-weight:600;margin-bottom:4px;">${data.method} · ${data.backbone || ''}</div>`;
    }
    if (data.acc !== undefined) {
      content += `<div>Acc: <strong>${(data.acc * 100).toFixed(1)}%</strong></div>`;
    }
    if (data.vram !== undefined) {
      content += `<div>Peak VRAM: <strong>${data.vram.toFixed(1)} GB</strong></div>`;
    }
    if (data.gei !== undefined) {
      content += `<div>GEI Score: <strong>${data.gei.toFixed(3)}</strong></div>`;
    }

    tooltipEl.innerHTML = content;
    tooltipEl.classList.add('is-visible');

    const clientX = evt.clientX || (evt.target && evt.target.getBoundingClientRect().right) || 100;
    const clientY = evt.clientY || (evt.target && evt.target.getBoundingClientRect().top) || 100;

    let x = clientX + 14;
    let y = clientY - 10;
    if (x + 220 > window.innerWidth) x = clientX - 230;
    if (y + 100 > window.innerHeight) y = clientY - 80;

    tooltipEl.style.left = `${Math.max(10, x)}px`;
    tooltipEl.style.top = `${Math.max(10, y)}px`;
  }

  function hideTooltip() {
    if (tooltipEl) {
      tooltipEl.classList.remove('is-visible');
    }
  }

  return {
    initHeroScatter: initHeroScatter,
    initEvidenceScatter: initEvidenceScatter,
    initFilterToggleSvg: initFilterToggleSvg,
    showTooltip: showTooltip,
    hideTooltip: hideTooltip,
    refreshTheme: function () {
      if (heroChartInstance && heroChartInstance.rebuild) heroChartInstance.rebuild();
      const evidenceMount = document.getElementById('evidence-scatter-mount');
      if (evidenceMount) initEvidenceScatter(evidenceMount);
      const filterMount = document.getElementById('filter-dots-mount');
      if (filterMount) initFilterToggleSvg(filterMount);
    }
  };
})();
