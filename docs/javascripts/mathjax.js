window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  setTimeout(() => {
    if (typeof MathJax !== 'undefined') {
      if (MathJax.startup) {
        MathJax.startup.defaultReady();
      }
      if (MathJax.typesetPromise) {
        MathJax.typesetPromise().catch(function (err) {
          console.log('MathJax typeset error:', err);
        });
      }
    }
  }, 100);
})
