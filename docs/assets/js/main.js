(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll(".slide"));
  var dotsWrap = document.getElementById("dots");
  var prevBtn = document.getElementById("prevBtn");
  var nextBtn = document.getElementById("nextBtn");
  var current = 0;

  slides.forEach(function (_, i) {
    var dot = document.createElement("button");
    dot.className = "dot";
    dot.setAttribute("aria-label", "Go to slide " + (i + 1));
    dot.addEventListener("click", function () { goTo(i); });
    dotsWrap.appendChild(dot);
  });
  var dots = Array.prototype.slice.call(dotsWrap.children);

  function goTo(index) {
    if (index < 0 || index >= slides.length) return;
    slides[current].classList.remove("active");
    dots[current].classList.remove("active");
    current = index;
    slides[current].classList.add("active");
    dots[current].classList.add("active");
    prevBtn.disabled = current === 0;
    nextBtn.disabled = current === slides.length - 1;
    history.replaceState(null, "", "#slide-" + (current + 1));
  }

  prevBtn.addEventListener("click", function () { goTo(current - 1); });
  nextBtn.addEventListener("click", function () { goTo(current + 1); });

  document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowRight" || e.key === "PageDown") goTo(current + 1);
    if (e.key === "ArrowLeft" || e.key === "PageUp") goTo(current - 1);
  });

  var startIndex = 0;
  var hashMatch = /^#slide-(\d+)$/.exec(location.hash);
  if (hashMatch) {
    var n = parseInt(hashMatch[1], 10) - 1;
    if (n >= 0 && n < slides.length) startIndex = n;
  }
  goTo(startIndex);
})();
