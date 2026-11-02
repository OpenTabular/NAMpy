(function () {
  var root = document.querySelector(".arch-atlas");
  if (!root) return;

  function escapeText(value) {
    return value.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  }

  function appendToken(fragment, token, kind) {
    var span = document.createElement("span");
    span.className = "tok-" + kind;
    span.textContent = token;
    fragment.appendChild(span);
  }

  function paintIdentifiers() {
    var functions = Object.create(null);
    functions.evaluate = 1;
    functions.architectures = 1;
    functions.register_architecture = 1;
    functions.get_architecture = 1;
    var classes = Object.create(null);

    function isClass(token) {
      if (Object.prototype.hasOwnProperty.call(classes, token)) return true;
      if (!/^[A-Z][A-Za-z0-9]*$/.test(token)) return false;
      if (!/[a-z]{2}/.test(token)) return false;
      return /[A-Z].*[A-Z]/.test(token);
    }

    root.querySelectorAll("code").forEach(function (code) {
      var text = code.textContent;
      var dotted = /[A-Z][A-Za-z0-9]*\.[A-Za-z_][A-Za-z0-9_]*/g;
      var found;
      while ((found = dotted.exec(text))) {
        classes[found[0].slice(0, found[0].indexOf("."))] = 1;
      }
      text.split(/[^A-Za-z0-9_]+/).forEach(function (part) {
        if (isClass(part)) classes[part] = 1;
      });
    });
    root.querySelectorAll(".arch-fn code").forEach(function (code) {
      code.textContent.split(/[^A-Za-z0-9_]+/).forEach(function (part) {
        if (part && !isClass(part)) functions[part] = 1;
      });
    });
    root.querySelectorAll("details").forEach(function (group) {
      var summary = group.querySelector("summary");
      if (!summary || !/methods|dispatches|adapter/i.test(summary.textContent)) return;
      group.querySelectorAll("code").forEach(function (code) {
        code.textContent.split(/[^A-Za-z0-9_]+/).forEach(function (part) {
          if (part && !isClass(part)) functions[part] = 1;
        });
      });
    });

    function kindOf(token, slash) {
      if (slash) return "mod";
      if (isClass(token)) return "cls";
      if (functions[token]) return "fn";
      return "var";
    }

    function tokenSpan(token, kind) {
      return '<span class="tok-' + kind + '">' + escapeText(token) + "</span>";
    }

    root.querySelectorAll("code").forEach(function (code) {
      var text = code.textContent;
      var html = "";
      var re = /[A-Z][A-Za-z0-9]*\.[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*\/?/g;
      var last = 0;
      var match;
      while ((match = re.exec(text))) {
        html += escapeText(text.slice(last, match.index));
        var raw = match[0];
        var dot = raw.indexOf(".");
        if (dot !== -1) {
          html += tokenSpan(raw.slice(0, dot), "cls") + "." + tokenSpan(raw.slice(dot + 1), "fn");
        } else {
          var slash = raw.charAt(raw.length - 1) === "/";
          var token = slash ? raw.slice(0, -1) : raw;
          html += tokenSpan(token, kindOf(token, slash));
          if (slash) html += "/";
        }
        last = match.index + raw.length;
      }
      html += escapeText(text.slice(last));
      code.innerHTML = html;
    });

    root.querySelectorAll(".arch-lead, .arch-note, .arch-row p, .arch-step p, summary").forEach(function (block) {
      var walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, null);
      var nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach(function (node) {
        var text = node.nodeValue;
        if (!text || !/[A-Za-z_]/.test(text)) return;
        var re = /[A-Z][A-Za-z0-9]*\.[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*/g;
        var match;
        var changed = false;
        var fragment = document.createDocumentFragment();
        var last = 0;
        while ((match = re.exec(text))) {
          var raw = match[0];
          var dot = raw.indexOf(".");
          var kind = null;
          if (dot !== -1 && isClass(raw.slice(0, dot))) {
            changed = true;
            if (match.index > last) fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
            appendToken(fragment, raw.slice(0, dot), "cls");
            fragment.appendChild(document.createTextNode("."));
            appendToken(fragment, raw.slice(dot + 1), "fn");
            last = match.index + raw.length;
            continue;
          }
          if (isClass(raw)) kind = "cls";
          else if (functions[raw] && raw.indexOf("_") !== -1) kind = "fn";
          else if (raw.indexOf("_") !== -1) kind = "var";
          if (!kind) continue;
          changed = true;
          if (match.index > last) fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
          appendToken(fragment, raw, kind);
          last = match.index + raw.length;
        }
        if (!changed) return;
        if (last < text.length) fragment.appendChild(document.createTextNode(text.slice(last)));
        node.parentNode.replaceChild(fragment, node);
      });
    });
  }

  paintIdentifiers();

  var buttons = Array.prototype.slice.call(root.querySelectorAll(".arch-nav button"));
  var panels = Array.prototype.slice.call(root.querySelectorAll(".arch-panel"));
  var search = root.querySelector(".arch-search");
  var clear = root.querySelector(".arch-clear");
  var empty = root.querySelector(".arch-empty");

  function panelFor(id) {
    return root.querySelector('.arch-panel[data-stage="' + id + '"]');
  }

  function show(id, updateHash) {
    buttons.forEach(function (button) {
      var on = button.getAttribute("data-stage") === id;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-selected", on ? "true" : "false");
    });
    panels.forEach(function (panel) {
      var on = panel.getAttribute("data-stage") === id;
      panel.classList.toggle("is-active", on);
      panel.hidden = !on;
    });
    var selected = root.querySelector('.arch-nav button[data-stage="' + id + '"]');
    if (selected) {
      var nav = selected.parentElement;
      if (nav.scrollWidth > nav.clientWidth + 4) {
        var navRect = nav.getBoundingClientRect();
        var buttonRect = selected.getBoundingClientRect();
        nav.scrollBy({
          left: buttonRect.left - navRect.left - (nav.clientWidth - buttonRect.width) / 2,
          behavior: "smooth"
        });
      }
    }
    if (updateHash) history.replaceState(null, "", "#arch-" + id);
  }

  function applyFilter(query) {
    var q = query.trim().toLowerCase();
    var first = null;
    if (clear) clear.hidden = !q;
    buttons.forEach(function (button) {
      var id = button.getAttribute("data-stage");
      var panel = panelFor(id);
      var pieces = panel.querySelectorAll(".arch-row, .arch-fn, .arch-step");
      var hits = 0;
      pieces.forEach(function (piece) {
        if (!q || piece.textContent.toLowerCase().indexOf(q) !== -1) hits += 1;
      });
      var lead = panel.querySelector(".arch-lead");
      var leadHit = !!(lead && lead.textContent.toLowerCase().indexOf(q) !== -1);
      pieces.forEach(function (piece) {
        var match = piece.textContent.toLowerCase().indexOf(q) !== -1;
        piece.classList.toggle("is-dim", !!q && hits > 0 && !match);
      });
      panel.querySelectorAll(".arch-group").forEach(function (group) {
        var visible = group.querySelectorAll(".arch-row:not(.is-dim), .arch-step:not(.is-dim)").length;
        group.hidden = !!q && hits > 0 && visible === 0;
        if (q && visible) group.open = true;
      });
      var relevant = !q || hits > 0 || leadHit;
      button.classList.toggle("is-muted", !!q && !relevant);
      if (relevant && !first) first = id;
    });
    var active = root.querySelector(".arch-nav button.is-active");
    if (!q) {
      show(active ? active.getAttribute("data-stage") : "lifecycle", false);
    } else if (active && active.classList.contains("is-muted") && first) {
      show(first, true);
    } else if (!first) {
      panels.forEach(function (panel) { panel.hidden = true; });
    }
    if (empty) empty.hidden = !q || !!first;
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      show(button.getAttribute("data-stage"), true);
      if (search && search.value.trim()) applyFilter(search.value);
    });
  });

  root.querySelectorAll("[data-goto]").forEach(function (button) {
    button.addEventListener("click", function () {
      show(button.getAttribute("data-goto"), true);
      var panel = panelFor(button.getAttribute("data-goto"));
      if (panel) panel.focus();
    });
  });

  if (search) search.addEventListener("input", function () { applyFilter(search.value); });
  if (clear) {
    clear.addEventListener("click", function () {
      search.value = "";
      applyFilter("");
      search.focus();
    });
  }

  var initial = "lifecycle";
  var hash = location.hash.replace("#arch-", "");
  if (panelFor(hash)) initial = hash;
  root.classList.add("is-ready");
  show(initial, false);
})();
