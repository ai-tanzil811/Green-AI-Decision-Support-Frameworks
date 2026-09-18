// GreenPEFT Interactive Simulator & Utility Script

document.addEventListener('DOMContentLoaded', () => {
  initRecommenderSimulator();
  initCopyButtons();
});

// Candidate configurations dataset based on empirical Kaggle Tesla T4 benchmark
const CANDIDATES = [
  {
    name: "QLoRA + Qwen2.5-3B",
    strategy: "QLoRA",
    backbone: "Qwen2.5-3B (3.0B params)",
    accuracy: 0.9484,
    vram: 7.45,
    energy: 0.00601,
    carbon: 0.00391,
    time: 520,
    quant: "4-bit (NF4)"
  },
  {
    name: "LoRA + Qwen2.5-1.5B",
    strategy: "LoRA",
    backbone: "Qwen2.5-1.5B (1.5B params)",
    accuracy: 0.9478,
    vram: 12.28,
    energy: 0.00218,
    carbon: 0.00142,
    time: 410,
    quant: "FP16"
  },
  {
    name: "QLoRA + Qwen2.5-1.5B",
    strategy: "QLoRA",
    backbone: "Qwen2.5-1.5B (1.5B params)",
    accuracy: 0.9434,
    vram: 6.78,
    energy: 0.00344,
    carbon: 0.00223,
    time: 380,
    quant: "4-bit (NF4)"
  },
  {
    name: "LoRA + TinyLlama-1.1B",
    strategy: "LoRA",
    backbone: "TinyLlama-1.1B (1.1B params)",
    accuracy: 0.9438,
    vram: 9.74,
    energy: 0.00177,
    carbon: 0.00115,
    time: 340,
    quant: "FP16"
  },
  {
    name: "LoRA + Qwen2.5-0.5B",
    strategy: "LoRA",
    backbone: "Qwen2.5-0.5B (0.5B params)",
    accuracy: 0.9174,
    vram: 4.65,
    energy: 0.00122,
    carbon: 0.00079,
    time: 190,
    quant: "FP16"
  },
  {
    name: "LISA + Qwen2.5-0.5B",
    strategy: "LISA",
    backbone: "Qwen2.5-0.5B (0.5B params)",
    accuracy: 0.9122,
    vram: 4.84,
    energy: 0.00324,
    carbon: 0.00211,
    time: 183,
    quant: "FP16"
  },
  {
    name: "QLoRA + Qwen2.5-0.5B",
    strategy: "QLoRA",
    backbone: "Qwen2.5-0.5B (0.5B params)",
    accuracy: 0.9117,
    vram: 3.07,
    energy: 0.00215,
    carbon: 0.00140,
    time: 210,
    quant: "4-bit (NF4)"
  },
  {
    name: "Full Fine-Tuning + Qwen2.5-0.5B",
    strategy: "Full-FT",
    backbone: "Qwen2.5-0.5B (0.5B params)",
    accuracy: 0.9122,
    vram: 13.48,
    energy: 0.00382,
    carbon: 0.00248,
    time: 223,
    quant: "FP16"
  }
];

// Preference Weight Profiles
const PROFILES = {
  balanced: { wAcc: 0.35, wMem: 0.25, wCarb: 0.25, wTime: 0.15, label: "Balanced Profile" },
  strict_carbon: { wAcc: 0.20, wMem: 0.20, wCarb: 0.50, wTime: 0.10, label: "Strict Carbon Profile" },
  high_accuracy: { wAcc: 0.60, wMem: 0.15, wCarb: 0.15, wTime: 0.10, label: "High-Accuracy Profile" }
};

let currentProfile = 'balanced';

function initRecommenderSimulator() {
  const vramSlider = document.getElementById('vram-slider');
  const vramValue = document.getElementById('vram-val');
  const accSlider = document.getElementById('acc-slider');
  const accValue = document.getElementById('acc-val');
  const profileBtns = document.querySelectorAll('.profile-btn');

  if (!vramSlider || !accSlider) return;

  vramSlider.addEventListener('input', (e) => {
    vramValue.textContent = `${e.target.value} GB`;
    runRecommendation();
  });

  accSlider.addEventListener('input', (e) => {
    const pct = parseFloat(e.target.value).toFixed(2);
    accValue.textContent = `${(pct * 100).toFixed(0)}%`;
    runRecommendation();
  });

  profileBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      profileBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentProfile = btn.getAttribute('data-profile');
      runRecommendation();
    });
  });

  // Initial calculation
  runRecommendation();
}

function runRecommendation() {
  const maxVram = parseFloat(document.getElementById('vram-slider').value);
  const minAcc = parseFloat(document.getElementById('acc-slider').value);
  const profile = PROFILES[currentProfile];

  // 1. Constraint Filtering
  const feasible = CANDIDATES.filter(c => c.vram <= maxVram && c.accuracy >= minAcc);

  const recTitle = document.getElementById('rec-title');
  const recAcc = document.getElementById('rec-acc');
  const recVram = document.getElementById('rec-vram');
  const recCarbon = document.getElementById('rec-carbon');
  const recGei = document.getElementById('rec-gei');
  const recDesc = document.getElementById('rec-desc');

  if (feasible.length === 0) {
    recTitle.textContent = "No Feasible Strategy Found";
    recAcc.textContent = "--";
    recVram.textContent = "--";
    recCarbon.textContent = "--";
    recGei.textContent = "0.000";
    recDesc.innerHTML = `<span style="color:#f43f5e;">⚠️ Active constraints (VRAM ≤ ${maxVram}GB, Accuracy ≥ ${(minAcc*100).toFixed(0)}%) are too restrictive. Try increasing VRAM limit or relaxing accuracy floor.</span>`;
    return;
  }

  // 2. Normalization & GEI Scoring
  const accs = feasible.map(c => c.accuracy);
  const vrams = feasible.map(c => c.vram);
  const carbs = feasible.map(c => c.carbon);
  const times = feasible.map(c => c.time);

  const minAccVal = Math.min(...accs), maxAccVal = Math.max(...accs);
  const minVramVal = Math.min(...vrams), maxVramVal = Math.max(...vrams);
  const minCarbVal = Math.min(...carbs), maxCarbVal = Math.max(...carbs);
  const minTimeVal = Math.min(...times), maxTimeVal = Math.max(...times);

  const scored = feasible.map(c => {
    const sAcc = maxAccVal === minAccVal ? 1 : (c.accuracy - minAccVal) / (maxAccVal - minAccVal);
    const sMem = maxVramVal === minVramVal ? 1 : 1 - (c.vram - minVramVal) / (maxVramVal - minVramVal);
    const sCarb = maxCarbVal === minCarbVal ? 1 : 1 - (c.carbon - minCarbVal) / (maxCarbVal - minCarbVal);
    const sTime = maxTimeVal === minTimeVal ? 1 : 1 - (c.time - minTimeVal) / (maxTimeVal - minTimeVal);

    const gei = (profile.wAcc * sAcc) + (profile.wMem * sMem) + (profile.wCarb * sCarb) + (profile.wTime * sTime);
    return { ...c, gei };
  });

  // 3. Rank Top Recommendation
  scored.sort((a, b) => b.gei - a.gei);
  const winner = scored[0];

  recTitle.textContent = winner.name;
  recAcc.textContent = `${(winner.accuracy * 100).toFixed(2)}%`;
  recVram.textContent = `${winner.vram} GB`;
  recCarbon.textContent = `${(winner.carbon * 1000).toFixed(2)} gCO₂`;
  recGei.textContent = winner.gei.toFixed(3);

  const savedCarbon = ((0.00391 - winner.carbon) / 0.00391 * 100).toFixed(0);

  recDesc.innerHTML = `Prescribed under <strong>${profile.label}</strong>. Satisfies peak VRAM constraint (${winner.vram} GB ≤ ${maxVram} GB) and accuracy floor (${(winner.accuracy*100).toFixed(1)}% ≥ ${(minAcc*100).toFixed(0)}%). Reduces carbon emissions by up to <strong>${savedCarbon > 0 ? savedCarbon : 0}%</strong> compared to max-size baseline.`;
}

// Copy to Clipboard Utility with Toast
function initCopyButtons() {
  const copyBtns = document.querySelectorAll('[data-copy]');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const textToCopy = btn.getAttribute('data-copy');
      navigator.clipboard.writeText(textToCopy).then(() => {
        showToast('Copied to clipboard!');
      });
    });
  });
}

function showToast(message) {
  let toast = document.getElementById('toast-notification');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast-notification';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
  }, 2500);
}
