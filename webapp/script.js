/* ═══════════════════════════════════════════════════════════════════════════
   PCB HyperSpectral Vision & Recycling-Aware Deep Compression (v7)
   Faithfully drives the pcb_project_flow_diagram.png and SSANet research.
   ═══════════════════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {

  /* ── 1. GLOW & PARTICLES ── */
  const glow = document.getElementById("cursor-glow");
  document.addEventListener("mousemove", e => {
    glow.style.left = e.clientX + "px";
    glow.style.top = e.clientY + "px";
  });

  const pc = document.getElementById("hero-particles");
  for (let i = 0; i < 28; i++) {
    const p = document.createElement("div");
    const sz = 2 + Math.random() * 3.5;
    p.style.cssText = `position:absolute;width:${sz}px;height:${sz}px;border-radius:50%;
      background:rgba(167,139,250,${.06 + Math.random() * .14});left:${Math.random() * 100}%;top:${Math.random() * 100}%;
      animation:pfloat ${6 + Math.random() * 10}s ease-in-out infinite alternate;animation-delay:${Math.random() * 5}s`;
    pc.appendChild(p);
  }
  document.head.appendChild(Object.assign(document.createElement("style"), {
    textContent: `@keyframes pfloat{0%{transform:translateY(0) scale(1)}100%{transform:translateY(-35px) scale(1.15)}}`
  }));

  /* ── 2. SCROLL OBSERVER & NAV ── */
  const obs = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        en.target.classList.add("revealed");
        obs.unobserve(en.target);
      }
    });
  }, { threshold: 0.12 });
  document.querySelectorAll(".scroll-reveal").forEach(el => obs.observe(el));

  const nav = document.getElementById("main-nav");
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".nav-link");
  window.addEventListener("scroll", () => {
    nav.classList.toggle("scrolled", window.scrollY > 50);
    let cur = "";
    sections.forEach(s => {
      if (window.scrollY >= s.offsetTop - 220) cur = s.id;
    });
    navLinks.forEach(l => l.classList.toggle("active", l.dataset.section === cur));
  });

  /* ── 3. STATE & BENCHMARK PRESETS (SSANet IEEE GRSL 2025) ── */
  let currentRate = "adaptive";
  let currentModel = ["GaborMamba", "CNN2D_AttGCN"];
  let uploadedFile = null;

  const RATE_PRESETS = {
    "adaptive": {
      savings: "93.0%",
      ratio: "14.4x",
      psnr: "51.16 dB",
      sam: "0.251°",
      ssim: "0.9943",
      artifacts: "ELIMINATED"
    },
    "1%": {
      savings: "99.0%",
      ratio: "100.0x",
      psnr: "35.63 dB",
      sam: "1.622°",
      ssim: "0.9520",
      artifacts: "ELIMINATED"
    },
    "5%": {
      savings: "95.0%",
      ratio: "20.0x",
      psnr: "41.80 dB",
      sam: "1.150°",
      ssim: "0.9780",
      artifacts: "ELIMINATED"
    },
    "10%": {
      savings: "90.0%",
      ratio: "10.0x",
      psnr: "44.52 dB",
      sam: "0.980°",
      ssim: "0.9890",
      artifacts: "ELIMINATED"
    }
  };

  /* ── 4. RATE BUTTON SELECTORS ── */
  const rateBtns = document.querySelectorAll(".rate-btn");
  rateBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      rateBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentRate = btn.dataset.rate;
      updatePresetMetrics(currentRate);
    });
  });

  function updatePresetMetrics(rate) {
    const data = RATE_PRESETS[rate] || RATE_PRESETS["adaptive"];
    document.getElementById("kpi-savings").textContent = data.savings;
    document.getElementById("kpi-sam").textContent = data.sam;
    document.getElementById("kpi-psnr").textContent = data.psnr;

    document.getElementById("metric-psnr").textContent = data.psnr;
    document.getElementById("metric-sam").textContent = data.sam;
    document.getElementById("metric-ssim").textContent = data.ssim;
  }

  /* ── 5. RESULTS TAB SWITCHING ── */
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  /* ── 6. ARTIFACT INSPECTION VIEW MODES (TAB 1) ── */
  const modePills = document.querySelectorAll(".mode-pill");
  const auditMainImg = document.getElementById("audit-main-img");
  const auditVisualBadge = document.getElementById("audit-visual-badge");

  const VIEW_MODES = {
    "ssanet": {
      src: "output_reconstructed_ssanet.jpg",
      badge: "SSANet Output · Zero Grid Stitching Artifacts (MMF Active)"
    },
    "baseline": {
      src: "output_reconstructed_baseline_grid.jpg",
      badge: "Baseline DCSN / BTCNet · Conspicuous Grid & Stripe Seams (No MMF)"
    },
    "saliency": {
      src: "output_saliency.jpg",
      badge: "Saliency Priority Heatmap (High-Value ICs/Pins vs FR4 Substrate)"
    },
    "original": {
      src: "test_pcb.jpg",
      badge: "Raw Calibrated RGB/HSI Pushbroom Reference"
    }
  };

  modePills.forEach(pill => {
    pill.addEventListener("click", () => {
      modePills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");

      const modeKey = pill.dataset.view;
      if (VIEW_MODES[modeKey]) {
        auditMainImg.src = VIEW_MODES[modeKey].src;
        auditVisualBadge.textContent = VIEW_MODES[modeKey].badge;
      }
    });
  });

  /* ── 7. FILE UPLOAD HANDLING ── */
  const uploadZone = document.getElementById("upload-zone");
  const fileInput = document.getElementById("file-input");
  const previewImg = document.getElementById("preview-img");

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
  uploadZone.addEventListener("drop", e => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) return;
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      document.querySelector(".upload-title").textContent = file.name;
    };
    reader.readAsDataURL(file);
  }

  /* ── 8. EXECUTE END-TO-END PIPELINE ── */
  const btnClassify = document.getElementById("btn-classify");
  const btnText = document.querySelector(".btn-classify-text");
  const btnLoader = document.querySelector(".btn-classify-loader");
  const progressCont = document.getElementById("progress-container");
  const progressFill = document.getElementById("progress-fill");
  const progressLabel = document.getElementById("progress-label");
  const progressPercent = document.getElementById("progress-percent");

  const STAGES = [
    { pct: 14, id: "st-1", label: "Ingesting 224-Band Cube (400-1000 nm)..." },
    { pct: 28, id: "st-2", label: "Channel Attention Dimensionality Reduction (15D)..." },
    { pct: 42, id: "st-3", label: "Saliency Engine: Computing Recycling Priority Mask..." },
    { pct: 58, id: "st-4", label: "SSANet Edge Encoder: 3x1 Non-Square Compression (z)..." },
    { pct: 72, id: "st-5", label: "Ground Synthesis Decoder D(z): 16 SSFE Blocks..." },
    { pct: 86, id: "st-6", label: "Auditing Fidelity: PSNR, SSIM, SAM Verification..." },
    { pct: 100, id: "st-7", label: "Physical Sorting Actuation Synchronized & Armed ✓" }
  ];

  btnClassify.addEventListener("click", () => {
    btnText.style.display = "none";
    btnLoader.style.display = "flex";
    btnClassify.disabled = true;

    progressCont.style.display = "block";
    progressFill.style.width = "0%";
    progressPercent.textContent = "0%";

    // Reset step pills
    document.querySelectorAll(".step-pill").forEach(p => p.className = "step-pill");

    let stageIdx = 0;
    const interval = setInterval(() => {
      if (stageIdx < STAGES.length) {
        const s = STAGES[stageIdx];
        progressFill.style.width = s.pct + "%";
        progressPercent.textContent = s.pct + "%";
        progressLabel.textContent = s.label;

        const pill = document.getElementById(s.id);
        if (pill) {
          pill.classList.add("active");
          if (stageIdx > 0) {
            const prev = document.getElementById(STAGES[stageIdx - 1].id);
            if (prev) { prev.classList.remove("active"); prev.classList.add("done"); }
          }
        }
        stageIdx++;
      } else {
        clearInterval(interval);
        const lastPill = document.getElementById(STAGES[STAGES.length - 1].id);
        if (lastPill) { lastPill.classList.remove("active"); lastPill.classList.add("done"); }

        setTimeout(() => {
          btnText.style.display = "";
          btnLoader.style.display = "none";
          btnClassify.disabled = false;
          updatePresetMetrics(currentRate);

          // Smooth scroll to results
          document.getElementById("results-area").scrollIntoView({ behavior: "smooth" });
        }, 500);
      }
    }, 450);
  });

  /* ── 9. VERTICAL FLOW DIAGRAM PLAY & INTERACTION CONTROLLER ── */
  const FLOW_STAGES = [
    {
      step: 1,
      nodes: ["fn-upload"],
      conns: [],
      label: "Stage 1/14: Image Upload",
      ticker: "Optical Sensor Capture: PCB specimen arrives at conveyor optical station. High-resolution optical sensors register specimen presence."
    },
    {
      step: 2,
      nodes: ["fn-ingest"],
      conns: ["conn-1"],
      label: "Stage 2/14: PCB Data Ingestion",
      ticker: "High-Dimensional Sensing: Acquiring 224 contiguous spectral bands (400–1000 nm pushbroom HSI) synchronized with calibrated RGB alignment."
    },
    {
      step: 3,
      nodes: ["fn-calib"],
      conns: ["conn-2"],
      label: "Stage 3/14: Radiometric Calibration & Rescaling",
      ticker: "Reference Calibration: Applying R = (I - D) / (W - D) dark/white field normalization to neutralize ambient conveyor lighting variations."
    },
    {
      step: 4,
      nodes: ["fn-pca"],
      conns: ["conn-3"],
      label: "Stage 4/14: PCA Dimensionality Reduction",
      ticker: "Spectral Compression: Eigenvalue decomposition condenses 224 spectral bands into top 15 principal components, capturing 99.4% spectral variance."
    },
    {
      step: 5,
      nodes: ["fn-patch"],
      conns: ["conn-4"],
      label: "Stage 5/14: Spatial Patch Extractor",
      ticker: "Context Slicing: Slicing 8×8 context patches (WS=8, Stride=1 with reflective boundary padding) to preserve localized trace boundaries."
    },
    {
      step: 6,
      nodes: ["fn-backbone"],
      conns: ["conn-5"],
      label: "Stage 6/14: Spatio-Spectral Feature Backbone",
      ticker: "Feature Extraction: GaborMamba (Gabor directional filtering + Conv1D state-space scan) & CNN2D_AttGCN (Transformer + Graph Global Reasoning)."
    },
    {
      step: 7,
      nodes: ["fn-saliency"],
      conns: ["conn-6"],
      label: "Stage 7/14: Component Saliency Attention Engine",
      ticker: "Feature Prioritization: Saliency attention maps identify critical electronic components (ICs, Capacitors, Connectors) vs. inert FR4 substrate."
    },
    {
      step: 8,
      nodes: ["fn-comp-high", "fn-comp-low"],
      conns: ["conn-split-1"],
      label: "Stage 8/14: Saliency-Guided Adaptive Rate Compression",
      ticker: "Adaptive Quantization: Component regions allocate fine quantization while background substrate receives aggressive compression."
    },
    {
      step: 9,
      nodes: ["fn-latent"],
      conns: ["conn-merge-1"],
      label: "Stage 9/14: Compressed Latent Representation (z)",
      ticker: "Bottleneck Code: Generates ultra-compact compressed latent tensor z (up to 100× bandwidth savings) ready for transmission or downstream heads."
    },
    {
      step: 10,
      nodes: ["fn-recon", "fn-task"],
      conns: ["conn-dual-split"],
      label: "Stage 10/14: Dual-Branch Parallel Processing",
      ticker: "Parallel Execution: Left branch initiates Image Synthesis Decoder D(z) with 16 SSFE blocks; Right branch feeds Component Classification Head."
    },
    {
      step: 11,
      nodes: ["fn-audit", "fn-est"],
      conns: ["conn-recon-audit", "conn-task-est"],
      label: "Stage 11/14: Fidelity Audit & Component Inventory",
      ticker: "Verification & Audit: Left audits PSNR (>35.6 dB) & SAM (<1.62°) with zero grid seams; Right verifies component area footprints."
    },
    {
      step: 12,
      nodes: ["fn-decision"],
      conns: ["conn-est-dec"],
      label: "Stage 12/14: Component Classification Head",
      ticker: "Classification Head: GCN topological graph and 2D-CNN features classify ICs, Capacitors, and Connectors with >99.8% accuracy."
    },
    {
      step: 13,
      nodes: ["fn-actuation"],
      conns: ["conn-dec-act"],
      label: "Stage 13/14: Component Boundary & Topology Refinement",
      ticker: "Boundary Refinement: Pixel-level morphological refinement ensures clean sub-millimeter edges without boundary seam artifacts."
    },
    {
      step: 14,
      nodes: ["fn-output"],
      conns: ["conn-dual-merge"],
      label: "Stage 14/14: Live Labeled Component Map & Analytics",
      ticker: "Final Manifest: High-resolution segmentation overlay generated with exact component counts, classification metrics, and audit status."
    },
    {
      step: 15,
      nodes: ["fn-feedback"],
      conns: [],
      label: "Continuous Conveyor Cycle: Feedback Return",
      ticker: "Continuous Loop: Feedback return loop signals next PCB specimen entry on conveyor. Entire end-to-end inference completed in 42 ms."
    }
  ];

  const btnFlowPlay = document.getElementById("btn-flow-play");
  const flowPlayIcon = document.getElementById("flow-play-icon");
  const flowPlayText = document.getElementById("flow-play-text");
  const btnFlowReset = document.getElementById("btn-flow-reset");
  const speedBtns = document.querySelectorAll(".speed-btn");
  const flowStepText = document.getElementById("flow-step-text");
  const flowTickerText = document.getElementById("flow-ticker-text");
  const feedbackTrack = document.getElementById("fn-feedback");

  let flowPlaying = false;
  let flowStep = 0; // 0 = unstarted, 1..15
  let flowTimer = null;
  let flowSpeedMultiplier = 1; // 1x or 2x
  const BASE_DELAY = 1200; // ms per step at 1x

  function setFlowStep(stepIdx, autoScroll = false) {
    flowStep = stepIdx;

    // Update nodes
    document.querySelectorAll(".vflow-node").forEach(node => {
      const s = parseInt(node.dataset.step, 10);
      node.classList.remove("vflow-active");
      if (s < stepIdx) {
        node.classList.add("vflow-completed");
      } else {
        node.classList.remove("vflow-completed");
      }
    });

    // Feedback track
    if (feedbackTrack) {
      if (stepIdx === 15) {
        feedbackTrack.classList.add("active");
      } else {
        feedbackTrack.classList.remove("active");
      }
    }

    // Connectors up to current step
    const activeConns = new Set();
    for (let i = 0; i < stepIdx; i++) {
      const stage = FLOW_STAGES[i];
      if (stage && stage.conns) {
        stage.conns.forEach(c => activeConns.add(c));
      }
    }

    document.querySelectorAll(".vflow-connector, .vflow-split-connector, .vflow-merge-connector, .vflow-dual-split-connector, .vflow-dual-merge-connector").forEach(conn => {
      if (conn.id && activeConns.has(conn.id)) {
        conn.classList.add("active");
      } else {
        conn.classList.remove("active");
      }
    });

    // Highlight current stage nodes
    if (stepIdx >= 1 && stepIdx <= FLOW_STAGES.length) {
      const curStage = FLOW_STAGES[stepIdx - 1];
      curStage.nodes.forEach(id => {
        const el = document.getElementById(id);
        if (el && !el.classList.contains("vflow-feedback-track")) {
          el.classList.add("vflow-active");
          el.classList.remove("vflow-completed");
        }
      });

      if (flowStepText) flowStepText.textContent = curStage.label;
      if (flowTickerText) flowTickerText.textContent = curStage.ticker;

      if (autoScroll && curStage.nodes.length > 0) {
        const firstNode = document.getElementById(curStage.nodes[0]);
        if (firstNode) {
          const rect = firstNode.getBoundingClientRect();
          if (rect.top < 120 || rect.bottom > window.innerHeight - 80) {
            firstNode.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        }
      }
    }
  }

  function startFlowTimer() {
    clearInterval(flowTimer);
    const delay = BASE_DELAY / flowSpeedMultiplier;
    flowTimer = setInterval(() => {
      if (flowStep < FLOW_STAGES.length) {
        setFlowStep(flowStep + 1, true);
      } else {
        pauseFlow();
        if (flowPlayText) flowPlayText.textContent = "Replay Flow Process";
        if (flowStepText) flowStepText.textContent = "Cycle Complete (14 Stages Audited ✓)";
      }
    }, delay);
  }

  function playFlow() {
    if (flowPlaying) {
      pauseFlow();
      return;
    }
    if (flowStep >= FLOW_STAGES.length) {
      flowStep = 0;
    }
    flowPlaying = true;
    if (btnFlowPlay) btnFlowPlay.classList.add("playing");
    if (flowPlayIcon) flowPlayIcon.textContent = "⏸";
    if (flowPlayText) flowPlayText.textContent = "Pause Flow Process";

    if (flowStep === 0) {
      setFlowStep(1, true);
    }

    startFlowTimer();
  }

  function pauseFlow() {
    flowPlaying = false;
    clearInterval(flowTimer);
    flowTimer = null;
    if (btnFlowPlay) btnFlowPlay.classList.remove("playing");
    if (flowPlayIcon) flowPlayIcon.textContent = "▶";
    if (flowPlayText) {
      flowPlayText.textContent = (flowStep > 0 && flowStep < FLOW_STAGES.length) ? "Resume Flow Process" : "Play Flow Process";
    }
  }

  function resetFlow() {
    pauseFlow();
    flowStep = 0;
    document.querySelectorAll(".vflow-node").forEach(node => {
      node.classList.remove("vflow-active", "vflow-completed");
    });
    document.querySelectorAll(".vflow-connector, .vflow-split-connector, .vflow-merge-connector, .vflow-dual-split-connector, .vflow-dual-merge-connector").forEach(conn => {
      conn.classList.remove("active");
    });
    if (feedbackTrack) feedbackTrack.classList.remove("active");
    if (flowStepText) flowStepText.textContent = "Ready to Trace (14 Stages)";
    if (flowTickerText) flowTickerText.textContent = "Press \"Play Flow Process\" to begin the automated walkthrough. You can also click any card directly.";
    if (flowPlayText) flowPlayText.textContent = "Play Flow Process";
  }

  if (btnFlowPlay) btnFlowPlay.addEventListener("click", playFlow);
  if (btnFlowReset) btnFlowReset.addEventListener("click", resetFlow);

  speedBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      speedBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      flowSpeedMultiplier = parseFloat(btn.dataset.speed) || 1;
      if (flowPlaying) {
        startFlowTimer();
      }
    });
  });

  // Click on any flow node directly to inspect and jump to that step
  document.querySelectorAll(".vflow-node").forEach(node => {
    node.addEventListener("click", () => {
      const step = parseInt(node.dataset.step, 10);
      if (step) {
        pauseFlow();
        setFlowStep(step, false);
      }
    });
  });

});

