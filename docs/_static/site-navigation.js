/* Preserve the document tree's position while navigating between pages. */
(() => {
  // Sphinx renders the active page and its ancestors. The theme's reset adds
  // an unconditional scrollIntoView(), even for a hash in the article.
  if (window.SphinxRtdTheme) {
    window.SphinxRtdTheme.Navigation.reset = () => {};
  }

  function initializeSiteNavigation() {
    const sidebar = document.querySelector('.wy-side-scroll');
    const menu = document.querySelector('.wy-menu-vertical');
    if (!sidebar || !menu) return;
    const storageKey = 'nampy-sidebar-navigation';

    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey) || 'null');
      if (saved && saved.destination === window.location.pathname) {
        // Keep Sphinx's current-page classes: restoring old branches as
        // 'current' makes previously visited pages look selected as well.
        sidebar.scrollTop = saved.scrollTop;
      }
    } catch (_) {
      // Storage may be unavailable in privacy modes; navigation still works.
    }

    menu.addEventListener('click', (event) => {
      if (event.target.closest('button')) return;
      const link = event.target.closest('a[href]');
      if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      const destination = new URL(link.href);
      if (destination.origin !== window.location.origin) return;
      try {
        sessionStorage.setItem(storageKey, JSON.stringify({
          destination: destination.pathname,
          scrollTop: sidebar.scrollTop,
        }));
      } catch (_) {
        // Do not block the link if browser storage is disabled.
      }
      // Keep the theme's delegated link handler from collapsing the tree just
      // before the next page paints. Native link navigation is preserved.
      event.stopPropagation();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSiteNavigation);
  } else {
    initializeSiteNavigation();
  }
})();
