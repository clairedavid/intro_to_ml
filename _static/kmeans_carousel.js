document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".km-carousel").forEach(function (root) {
    const slides = root.querySelectorAll(".km-slide");
    const prev = root.querySelector(".km-prev"), next = root.querySelector(".km-next");
    const dots = root.querySelector(".km-dots");
    let i = 0;
    slides.forEach(function (_, k) {
      const d = document.createElement("span");
      d.className = "km-dot";
      d.onclick = function () { show(k); };
      dots.appendChild(d);
    });
    function show(k) {
      i = Math.max(0, Math.min(slides.length - 1, k));
      slides.forEach(function (s, n) { s.classList.toggle("active", n === i); });
      dots.querySelectorAll(".km-dot").forEach(function (d, n) { d.classList.toggle("active", n === i); });
      prev.disabled = i === 0;
      next.disabled = i === slides.length - 1;
    }
    prev.onclick = function () { show(i - 1); };
    next.onclick = function () { show(i + 1); };
    root.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") show(i - 1);
      if (e.key === "ArrowRight") show(i + 1);
    });
    show(0);
  });
});
