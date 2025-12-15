window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    autoload: {
      color: [],
      colorv2: ['color']
    },
    packages: {'[+]': ['ams', 'newcommand', 'configmacros']}
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      MathJax.startup.document.inputJax[0].preFilters.add(({math}) => {
        if (math.display === false) {
          math.math = '\\(' + math.math + '\\)';
        }
      });
    }
  }
};

