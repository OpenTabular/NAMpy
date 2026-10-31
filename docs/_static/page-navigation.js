/* Page sections belong here; the theme's left navigation lists documents. */
(() => {
  function initializePageNavigation() {
    const article = document.querySelector('[itemprop="articleBody"]');
    const panel = document.querySelector('.page-navigation');
    if (!article || !panel) return;

    const sections = Array.from(article.querySelectorAll('h2, h3'))
      .map((heading) => {
        const anchor = heading.querySelector('a.headerlink');
        if (!anchor) return null;
        const label = heading.cloneNode(true);
        label.querySelectorAll('a.headerlink').forEach((node) => node.remove());
        return { heading, href: anchor.getAttribute('href'), label: label.textContent.trim() };
      })
      .filter(Boolean);
    if (sections.length < 2) return;

    const list = document.createElement('ul');
    let parentItem = null;
    let subsections = null;
    const links = sections.map((section) => {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = section.href;
      link.textContent = section.label;
      item.appendChild(link);
      if (section.heading.tagName === 'H3' && parentItem) {
        if (!subsections) {
          subsections = document.createElement('ul');
          parentItem.appendChild(subsections);
        }
        subsections.appendChild(item);
      } else {
        list.appendChild(item);
        parentItem = item;
        subsections = null;
      }
      return link;
    });
    panel.querySelector('nav').appendChild(list);

    // Keep the same sections available without the desktop right-hand column.
    const mobile = document.createElement('details');
    mobile.className = 'page-navigation-mobile';
    const summary = document.createElement('summary');
    summary.id = 'page-navigation-mobile-title';
    summary.textContent = 'On this page';
    const mobileNavigation = document.createElement('nav');
    mobileNavigation.setAttribute('aria-labelledby', summary.id);
    const mobileList = list.cloneNode(true);
    mobileNavigation.appendChild(mobileList);
    mobile.append(summary, mobileNavigation);
    const title = article.querySelector('h1');
    if (title) title.insertAdjacentElement('afterend', mobile);
    else article.prepend(mobile);
    const mobileLinks = Array.from(mobileList.querySelectorAll('a'));

    panel.hidden = false;
    panel.parentElement.classList.add('has-page-navigation');

    let scheduled = false;
    function updateActiveSection() {
      let current = 0;
      sections.forEach((section, index) => {
        if (section.heading.getBoundingClientRect().top <= 112) current = index;
      });
      if (window.scrollY > 0 &&
          Math.ceil(window.scrollY + window.innerHeight) >= document.documentElement.scrollHeight - 2) {
        current = sections.length - 1;
      }
      links.forEach((link, index) => {
        [link, mobileLinks[index]].forEach((item) => {
          if (index === current) item.setAttribute('aria-current', 'location');
          else item.removeAttribute('aria-current');
        });
      });
      scheduled = false;
    }
    function scheduleUpdate() {
      if (!scheduled) {
        scheduled = true;
        window.requestAnimationFrame(updateActiveSection);
      }
    }
    window.addEventListener('scroll', scheduleUpdate, { passive: true });
    window.addEventListener('resize', scheduleUpdate);
    window.addEventListener('hashchange', scheduleUpdate);
    updateActiveSection();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePageNavigation);
  } else {
    initializePageNavigation();
  }
})();
