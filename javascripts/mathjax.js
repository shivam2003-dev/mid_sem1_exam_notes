window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams'
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    ready: function() {
      MathJax.startup.defaultReady();
      // Re-typeset when Material theme switches pages
      if (typeof document$ !== 'undefined') {
        document$.subscribe(function() {
          MathJax.typesetPromise();
        });
      }
    }
  }
};
