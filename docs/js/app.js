const QUICK_SEARCHES = [
  { label: "Housing", query: "emergency housing assistance" },
  { label: "Safety", query: "domestic violence support" },
  { label: "Legal", query: "legal aid services" },
  { label: "Food", query: "food assistance" },
  { label: "Childcare", query: "childcare services" },
  { label: "Health", query: "healthcare services" },
  { label: "Jobs", query: "employment assistance" },
  { label: "Education", query: "education support" },
];

const state = {
  payload: null,
  fuse: null,
  lastResults: [],
};

function getBasePath() {
  const segments = window.location.pathname.split("/").filter(Boolean);
  if (segments.length === 0 || segments[0] === "index.html") {
    return "";
  }
  return `/${segments[0]}`;
}

function assetUrl(relativePath) {
  const base = getBasePath();
  return `${base}/${relativePath}`.replace(/\/{2,}/g, "/");
}

function formatType(value) {
  if (!value) return "general";
  return value.replace(/_/g, " ");
}

function truncate(text, max = 300) {
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function scoreFromFuse(result) {
  if (typeof result.score !== "number") {
    return null;
  }
  return Math.max(0, Math.min(1, 1 - result.score));
}

function applyPriorityBoost(resource, score) {
  let boosted = score ?? 0.5;
  const tier = resource.priority_tier ?? 3;

  if (tier === 0) boosted += 0.15;
  else if (tier === 1) boosted += 0.14;
  else if (tier === 2) boosted += 0.13;

  return Math.min(boosted, 1);
}

function sortResults(results) {
  return [...results].sort((a, b) => {
    const tierDiff = (a.priority_tier ?? 3) - (b.priority_tier ?? 3);
    if (tierDiff !== 0) return tierDiff;
    return (b.match_score ?? 0) - (a.match_score ?? 0);
  });
}

function renderStats(stats) {
  document.getElementById("stat-total").textContent = stats.total_resources;
  document.getElementById("stat-current").textContent = stats.current_resources;
  document.getElementById("stat-categories").textContent = stats.categories;
}

function renderQuickSearch() {
  const container = document.getElementById("quick-search");
  container.innerHTML = QUICK_SEARCHES.map(
    ({ label, query }, index) =>
      `<button type="button" class="quick-btn" data-query="${query}" data-index="${index}">${label}</button>`
  ).join("");

  container.addEventListener("click", (event) => {
    const button = event.target.closest("[data-query]");
    if (!button) return;
    document.getElementById("search-input").value = button.dataset.query;
    runSearch();
  });
}

function renderContact(resource) {
  const items = [];
  if (resource.phone) items.push(`Phone: ${resource.phone}`);
  if (resource.email) items.push(`Email: ${resource.email}`);
  if (resource.website) {
    items.push(
      `Website: <a href="${resource.website}" target="_blank" rel="noopener noreferrer">${resource.website}</a>`
    );
  }
  if (resource.address) items.push(`Address: ${resource.address}`);
  if (resource.hours) items.push(`Hours: ${resource.hours}`);

  if (!items.length) {
    return `<div class="contact-item">Contact information not available</div>`;
  }

  return items.map((item) => `<div class="contact-item">${item}</div>`).join("");
}

function renderResults(results) {
  const section = document.getElementById("results-section");
  const empty = document.getElementById("empty-state");
  const container = document.getElementById("results");
  const count = document.getElementById("results-count");

  if (!results.length) {
    section.hidden = true;
    empty.hidden = false;
    container.innerHTML = "";
    count.textContent = "";
    return;
  }

  section.hidden = false;
  empty.hidden = true;
  count.textContent = `Found ${results.length} resources`;

  container.innerHTML = results
    .map((resource) => {
      const isCcsf = resource.is_ccsf;
      const matchScore =
        typeof resource.match_score === "number"
          ? `<span class="match-score">${Math.round(resource.match_score * 100)}% match</span>`
          : "";
      const status = resource.is_current
        ? `<span class="status-current">Current</span>`
        : `<span class="status-outdated">May be outdated</span>`;

      return `
        <article class="resource-card ${isCcsf ? "ccsf" : ""}">
          <div class="resource-header">
            <h3 class="resource-title">${isCcsf ? "CCSF " : ""}${resource.organization_name || "Unknown Organization"}</h3>
            <div class="badges">
              <span class="badge">${formatType(resource.resource_type)}</span>
              ${isCcsf ? '<span class="badge ccsf">CCSF Campus</span>' : ""}
              ${resource.is_crisis_resource ? '<span class="badge crisis">Crisis support</span>' : ""}
            </div>
          </div>
          ${resource.description ? `<p class="resource-description">${truncate(resource.description)}</p>` : ""}
          <strong>Contact Information</strong>
          <div class="contact-list">${renderContact(resource)}</div>
          ${resource.eligibility ? `<div class="eligibility"><strong>Eligibility:</strong> ${truncate(resource.eligibility, 200)}</div>` : ""}
          <div class="meta-row">${matchScore}${status}</div>
        </article>
      `;
    })
    .join("");
}

function renderBrowseTable(resources) {
  const table = document.getElementById("resources-table").querySelector("tbody");
  table.innerHTML = resources
    .map(
      (resource) => `
      <tr>
        <td>${resource.organization_name || ""}</td>
        <td>${formatType(resource.resource_type)}</td>
        <td>${resource.phone || ""}</td>
        <td>${resource.address || ""}</td>
        <td>${resource.is_current ? "Yes" : "No"}</td>
      </tr>
    `
    )
    .join("");
}

function buildCsv(resources) {
  const headers = [
    "organization_name",
    "resource_type",
    "phone",
    "email",
    "website",
    "address",
    "hours",
    "is_current",
  ];

  const escape = (value) => `"${String(value ?? "").replace(/"/g, '""')}"`;
  const rows = resources.map((resource) =>
    headers.map((header) => escape(resource[header])).join(",")
  );

  return [headers.join(","), ...rows].join("\n");
}

function getFilteredResources() {
  const currentOnly = document.getElementById("current-only").checked;
  return state.payload.resources.filter((resource) =>
    currentOnly ? resource.is_current : true
  );
}

function runSearch() {
  const query = document.getElementById("search-input").value.trim();
  const limit = Number(document.getElementById("result-count").value);
  const pool = getFilteredResources();

  if (!query) {
    state.lastResults = [];
    renderResults([]);
    return;
  }

  const fuse = new Fuse(pool, {
    keys: ["organization_name", "resource_type", "description", "services", "eligibility", "address", "categories", "search_text"],
    threshold: 0.4,
    ignoreLocation: true,
    includeScore: true,
  });

  const rawResults = fuse.search(query, { limit }).map((result) => {
    const matchScore = applyPriorityBoost(result.item, scoreFromFuse(result));
    return { ...result.item, match_score: matchScore };
  });

  state.lastResults = sortResults(rawResults).slice(0, limit);
  renderResults(state.lastResults);
}

function setupBrowseTable() {
  const toggle = document.getElementById("show-all");
  const tableSection = document.getElementById("browse-table");
  const downloadButton = document.getElementById("download-csv");

  toggle.addEventListener("change", () => {
    tableSection.hidden = !toggle.checked;
    if (toggle.checked) {
      renderBrowseTable(getFilteredResources());
    }
  });

  document.getElementById("current-only").addEventListener("change", () => {
    if (toggle.checked) {
      renderBrowseTable(getFilteredResources());
    }
    if (document.getElementById("search-input").value.trim()) {
      runSearch();
    }
  });

  downloadButton.addEventListener("click", () => {
    const csv = buildCsv(getFilteredResources());
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "wrc_resources.csv";
    link.click();
    URL.revokeObjectURL(url);
  });
}

async function init() {
  renderQuickSearch();
  setupBrowseTable();

  document.getElementById("search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch();
  });

  const response = await fetch(assetUrl("data/resources.json"));
  if (!response.ok) {
    throw new Error("Could not load resource data.");
  }

  state.payload = await response.json();
  renderStats(state.payload.stats);

  const params = new URLSearchParams(window.location.search);
  const initialQuery = params.get("q");
  if (initialQuery) {
    document.getElementById("search-input").value = initialQuery;
    runSearch();
  }
}

init().catch((error) => {
  document.querySelector(".page").innerHTML = `
    <section class="panel">
      <h2>Unable to load resources</h2>
      <p>${error.message}</p>
      <p>Make sure GitHub Pages is enabled and the data export has been generated.</p>
    </section>
  `;
});
