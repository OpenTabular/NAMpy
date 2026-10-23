/* Keep the pre-rendered contributor list current; retain it if GitHub is unavailable. */
(async () => {
  const container = document.getElementById("github-contributors");
  if (!container) return;
  try {
    const contributors = [];
    for (let page = 1; ; page += 1) {
      const response = await fetch(
        `https://api.github.com/repos/OpenTabular/NAMpy/contributors?per_page=100&page=${page}`,
        { headers: { Accept: "application/vnd.github+json" }, signal: AbortSignal.timeout(10000) }
      );
      if (!response.ok) return;
      const batch = await response.json();
      if (!Array.isArray(batch)) return;
      contributors.push(...batch);
      if (batch.length < 100) break;
    }
    const profiles = contributors.filter(person => /^[a-z\d-]+(?:\[bot\])?$/i.test(person.login));
    if (!profiles.length) return;
    const fragment = document.createDocumentFragment();
    for (const person of profiles) {
      const link = document.createElement("a");
      link.className = "contributor-profile";
      link.href = `https://github.com/${encodeURIComponent(person.login)}`;
      const avatar = document.createElement("img");
      const avatarURL = new URL(person.avatar_url);
      if (avatarURL.protocol !== "https:" || avatarURL.hostname !== "avatars.githubusercontent.com") continue;
      avatarURL.searchParams.set("s", "96");
      avatar.src = avatarURL.href;
      avatar.alt = "";
      avatar.width = avatar.height = 64;
      avatar.loading = "lazy";
      const name = document.createElement("span");
      name.textContent = person.login;
      link.append(avatar, name);
      fragment.append(link);
    }
    if (fragment.childElementCount) container.replaceChildren(fragment);
  } catch {
    // The initial profiles remain usable offline or when the API rate limit is reached.
  }
})();
