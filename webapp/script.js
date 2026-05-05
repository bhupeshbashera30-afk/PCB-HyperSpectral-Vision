/* ═══════════════════════════════════════════════════════════════
   PCB HyperSpectral Vision — v6: Static output + Lightbox
   Shows pre-generated detection images with click-to-enlarge
   ═══════════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {

  /* ── UI setup ── */
  const glow = document.getElementById("cursor-glow");
  document.addEventListener("mousemove", e => { glow.style.left = e.clientX+"px"; glow.style.top = e.clientY+"px"; });

  const obs = new IntersectionObserver(entries => {
    entries.forEach(en => { if (en.isIntersecting) { en.target.classList.add("revealed"); obs.unobserve(en.target); } });
  }, { threshold: 0.15 });
  document.querySelectorAll(".scroll-reveal").forEach(el => obs.observe(el));

  const nav = document.getElementById("main-nav");
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".nav-link");
  window.addEventListener("scroll", () => {
    nav.classList.toggle("scrolled", window.scrollY > 60);
    let cur = "";
    sections.forEach(s => { if (window.scrollY >= s.offsetTop - 200) cur = s.id; });
    navLinks.forEach(l => l.classList.toggle("active", l.dataset.section === cur));
  });

  const pc = document.getElementById("hero-particles");
  for (let i = 0; i < 25; i++) {
    const p = document.createElement("div");
    const sz = 2 + Math.random() * 3;
    p.style.cssText = `position:absolute;width:${sz}px;height:${sz}px;border-radius:50%;
      background:rgba(167,139,250,${.08+Math.random()*.12});left:${Math.random()*100}%;top:${Math.random()*100}%;
      animation:pfloat ${5+Math.random()*10}s ease-in-out infinite alternate;animation-delay:${Math.random()*5}s`;
    pc.appendChild(p);
  }
  document.head.appendChild(Object.assign(document.createElement("style"), {
    textContent: `@keyframes pfloat{0%{transform:translateY(0)}100%{transform:translateY(-30px)}}`
  }));

  const tooltip = document.getElementById("tooltip");
  document.querySelectorAll("[data-tooltip]").forEach(el => {
    el.addEventListener("mouseenter", () => { tooltip.textContent = el.dataset.tooltip; tooltip.classList.add("visible"); });
    el.addEventListener("mousemove", e => { tooltip.style.left=(e.clientX+14)+"px"; tooltip.style.top=(e.clientY+14)+"px"; });
    el.addEventListener("mouseleave", () => tooltip.classList.remove("visible"));
  });

  /* ── File upload ── */
  const uploadZone = document.getElementById("upload-zone");
  const uploadPreview = document.getElementById("upload-preview");
  const previewImg = document.getElementById("preview-img");
  const fileInput = document.getElementById("file-input");
  const btnRemove = document.getElementById("btn-remove");
  const btnClassify = document.getElementById("btn-classify");
  let uploadedImage = null;

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
  uploadZone.addEventListener("drop", e => { e.preventDefault(); uploadZone.classList.remove("drag-over"); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
  fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => { uploadedImage = img; previewImg.src = e.target.result; uploadZone.style.display = "none"; uploadPreview.style.display = "flex"; updateBtn(); };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  btnRemove.addEventListener("click", () => {
    uploadedImage = null; fileInput.value = ""; previewImg.src = "";
    uploadZone.style.display = ""; uploadPreview.style.display = "none";
    document.getElementById("results-area").style.display = "none";
    document.getElementById("progress-container").style.display = "none";
    updateBtn();
  });

  document.querySelectorAll('.model-toggle input').forEach(cb => cb.addEventListener("change", updateBtn));
  function getSelectedModels() { return Array.from(document.querySelectorAll('.model-toggle input:checked')).map(c => c.value); }
  function updateBtn() { btnClassify.disabled = !uploadedImage || getSelectedModels().length === 0; }

  /* ── Output images per model ── */
  const OUTPUT_IMAGES = {
    "GaborMamba": "output_result.jpg",
    "CNN2D_AttGCN": "output_result.jpg"
  };

  /* ── Lightbox ── */
  const lightbox = document.createElement("div");
  lightbox.className = "lightbox";
  lightbox.innerHTML = `<div class="lightbox-backdrop"></div><div class="lightbox-content"><img class="lightbox-img" src="" alt="Enlarged view"><div class="lightbox-close">&times;</div></div>`;
  document.body.appendChild(lightbox);

  const lbImg = lightbox.querySelector(".lightbox-img");
  lightbox.querySelector(".lightbox-backdrop").addEventListener("click", closeLightbox);
  lightbox.querySelector(".lightbox-close").addEventListener("click", closeLightbox);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeLightbox(); });

  function openLightbox(src) {
    lbImg.src = src;
    lightbox.classList.add("active");
    document.body.style.overflow = "hidden";
  }
  function closeLightbox() {
    lightbox.classList.remove("active");
    document.body.style.overflow = "";
  }

  /* ── Run classification ── */
  btnClassify.addEventListener("click", () => {
    if (btnClassify.disabled || !uploadedImage) return;
    const models = getSelectedModels();
    const btnText = document.querySelector(".btn-classify-text");
    const btnLoader = document.querySelector(".btn-classify-loader");
    const progressCont = document.getElementById("progress-container");
    const progressFill = document.getElementById("progress-fill");
    const progressLabel = document.getElementById("progress-label");

    btnText.style.display = "none"; btnLoader.style.display = "flex";
    btnClassify.disabled = true; progressCont.style.display = "block"; progressFill.style.width = "0%";

    const steps = [
      { pct: 15, label: "Loading hyperspectral data..." },
      { pct: 35, label: "Applying PCA reduction to 15 components..." },
      { pct: 55, label: "Extracting 8×8 spatial patches..." },
      { pct: 75, label: `Running ${models.join(" & ")} inference...` },
      { pct: 90, label: "Building component map..." },
      { pct: 100, label: "Complete ✓" }
    ];
    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx < steps.length) {
        progressFill.style.width = steps[stepIdx].pct + "%";
        progressLabel.textContent = steps[stepIdx].label;
        stepIdx++;
      } else {
        clearInterval(interval);
        setTimeout(() => {
          btnText.style.display = ""; btnLoader.style.display = "none";
          progressCont.style.display = "none"; updateBtn();
          showResults(models);
        }, 400);
      }
    }, 500);
  });

  function showResults(models) {
    const resultsArea = document.getElementById("results-area");
    const resultsGrid = document.getElementById("results-grid");
    resultsGrid.innerHTML = "";
    resultsArea.style.display = "block";

    models.forEach((modelName, mIdx) => {
      const isGreen = modelName === "CNN2D_AttGCN";
      const imgSrc = OUTPUT_IMAGES[modelName];

      const card = document.createElement("div");
      card.className = "result-card";
      card.style.animationDelay = (mIdx * 0.15) + "s";

      card.innerHTML = `
        <div class="result-header">
          <span class="result-dot" style="background:${isGreen?'#34d399':'#a78bfa'}"></span>
          <span class="result-name">${modelName}</span>
          <span class="result-count">Click image to enlarge</span>
        </div>
        <div class="result-img-wrap">
          <img src="${imgSrc}" alt="${modelName} detection result" class="result-output-img">
        </div>`;

      // Click to enlarge
      card.querySelector(".result-img-wrap").addEventListener("click", () => {
        openLightbox(imgSrc);
      });

      resultsGrid.appendChild(card);
    });

    setTimeout(() => resultsArea.scrollIntoView({ behavior: "smooth", block: "start" }), 300);
  }

  /* ── Smooth scroll ── */
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener("click", e => {
      e.preventDefault();
      const t = document.querySelector(a.getAttribute("href"));
      if (t) t.scrollIntoView({ behavior: "smooth" });
    });
  });
});
