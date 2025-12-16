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

// Process MathJax when document is ready
if (typeof document$ !== 'undefined') {
  document$.subscribe(() => {
    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  });
} else {
  // Fallback if document$ is not available
  window.addEventListener('DOMContentLoaded', function() {
    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  });
  
  // Also try after page load
  window.addEventListener('load', function() {
    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
      setTimeout(() => {
        MathJax.typesetPromise();
      }, 500);
    }
  });
}
