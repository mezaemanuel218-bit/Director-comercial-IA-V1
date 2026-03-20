const healthStatus = document.getElementById("healthStatus");
const healthDetail = document.getElementById("healthDetail");
const loginOverlay = document.getElementById("loginOverlay");
const loginUsername = document.getElementById("loginUsername");
const loginPassword = document.getElementById("loginPassword");
const loginButton = document.getElementById("loginButton");
const loginMessage = document.getElementById("loginMessage");
const sessionUser = document.getElementById("sessionUser");
const sessionDetail = document.getElementById("sessionDetail");
const logoutButton = document.getElementById("logoutButton");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const refreshButton = document.getElementById("refreshButton");
const answerOutput = document.getElementById("answerOutput");
const modePill = document.getElementById("modePill");
const sourcePill = document.getElementById("sourcePill");
const webPill = document.getElementById("webPill");
const feedbackBanner = document.getElementById("feedbackBanner");
const refreshOutput = document.getElementById("refreshOutput");
const ownerFilter = document.getElementById("ownerFilter");
const dateFromFilter = document.getElementById("dateFromFilter");
const dateToFilter = document.getElementById("dateToFilter");
const statusFilter = document.getElementById("statusFilter");
const applyFiltersButton = document.getElementById("applyFiltersButton");
const clearFiltersButton = document.getElementById("clearFiltersButton");
const totalInteractions = document.getElementById("totalInteractions");
const activityTypes = document.getElementById("activityTypes");
const pendingCount = document.getElementById("pendingCount");
const staleCount = document.getElementById("staleCount");
const ownerLoadList = document.getElementById("ownerLoadList");
const pendingTasksList = document.getElementById("pendingTasksList");
const staleContactsList = document.getElementById("staleContactsList");
const recentActivityList = document.getElementById("recentActivityList");
const historyList = document.getElementById("historyList");
const prioritiesTable = document.getElementById("prioritiesTable");

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    healthStatus.textContent = data.status === "ok" ? "Conectada" : "Revisar";
    healthDetail.textContent = "API local lista para responder y refrescar información.";
  } catch (error) {
    healthStatus.textContent = "Sin conexión";
    healthDetail.textContent = "No fue posible conectar con la API local.";
  }
}

async function fetchCurrentUser() {
  try {
    const response = await fetch("/me");
    if (!response.ok) {
      throw new Error("Sin sesión");
    }
    const data = await response.json();
    sessionUser.textContent = data.display_name;
    sessionDetail.textContent = `Usuario activo: ${data.username}`;
    logoutButton.classList.remove("hidden");
    loginOverlay.classList.add("hidden");
    return data;
  } catch (error) {
    sessionUser.textContent = "Sin sesión";
    sessionDetail.textContent = "Inicia sesión para usar la plataforma.";
    logoutButton.classList.add("hidden");
    loginOverlay.classList.remove("hidden");
    return null;
  }
}

async function login() {
  const username = loginUsername.value.trim();
  const password = loginPassword.value;
  if (!username || !password) {
    loginMessage.textContent = "Ingresa usuario y contraseña.";
    return;
  }

  loginButton.disabled = true;
  loginButton.textContent = "Entrando...";
  loginMessage.textContent = "";

  try {
    const response = await fetch("/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, password }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "No se pudo iniciar sesión.");
    }
    loginPassword.value = "";
    await fetchCurrentUser();
    await Promise.all([loadOwners(), loadDashboard(), loadHistory()]);
  } catch (error) {
    loginMessage.textContent = error.message || "No se pudo iniciar sesión.";
  } finally {
    loginButton.disabled = false;
    loginButton.textContent = "Entrar";
  }
}

async function logout() {
  await fetch("/logout", { method: "POST" });
  await fetchCurrentUser();
}

function setBanner(message, isError = false) {
  feedbackBanner.textContent = message;
  feedbackBanner.classList.remove("hidden", "error");
  if (isError) {
    feedbackBanner.classList.add("error");
  }
}

function clearBanner() {
  feedbackBanner.classList.add("hidden");
  feedbackBanner.classList.remove("error");
  feedbackBanner.textContent = "";
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    setBanner("Escribe una pregunta antes de consultar.");
    questionInput.focus();
    return;
  }

  clearBanner();
  askButton.disabled = true;
  askButton.textContent = "Consultando...";
  answerOutput.textContent = "Pensando y reuniendo evidencia...";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    if (!response.ok) {
      if (response.status === 401) {
        await fetchCurrentUser();
      }
      throw new Error(data.detail || "No se pudo procesar la pregunta.");
    }

    answerOutput.textContent = data.answer;
    modePill.textContent = `Modo: ${data.mode}`;
    sourcePill.textContent = `Fuentes: ${data.sources.join(", ") || "-"}`;
    webPill.textContent = `Web: ${data.used_web ? "sí" : "no"}`;
    await loadHistory();
  } catch (error) {
    answerOutput.textContent = "No se pudo obtener una respuesta.";
    setBanner(error.message || "Ocurrió un error al consultar.", true);
  } finally {
    askButton.disabled = false;
    askButton.textContent = "Preguntar";
  }
}

async function refreshData() {
  clearBanner();
  refreshButton.disabled = true;
  refreshButton.textContent = "Actualizando...";
  refreshOutput.textContent = "Sincronizando Zoho, reconstruyendo warehouse e indexando PDFs...";

  try {
    const response = await fetch("/refresh", {
      method: "POST",
    });
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 401) {
        await fetchCurrentUser();
      }
      throw new Error(data.detail || "No se pudo actualizar la información.");
    }

    refreshOutput.innerHTML = `
      <strong>Módulos sincronizados:</strong> ${data.synced_modules.join(", ")}<br>
      <strong>Warehouse:</strong> leads ${data.warehouse.leads}, contacts ${data.warehouse.contacts}, notes ${data.warehouse.notes}, calls ${data.warehouse.calls}, tasks ${data.warehouse.tasks}, events ${data.warehouse.events}, interactions ${data.warehouse.interactions}<br>
      <strong>Documentos indexados:</strong> ${data.indexed_documents}
    `;
    setBanner("Actualización completada correctamente.");
    await loadDashboard();
  } catch (error) {
    refreshOutput.textContent = "La actualización no se completó.";
    setBanner(error.message || "Ocurrió un error al refrescar.", true);
  } finally {
    refreshButton.disabled = false;
    refreshButton.textContent = "Actualizar Zoho ahora";
  }
}

async function loadOwners() {
  try {
    const response = await fetch("/owners");
    if (response.status === 401) {
      return;
    }
    const data = await response.json();
    ownerFilter.innerHTML = '<option value="">Todos</option>';
    data.forEach((owner) => {
      const option = document.createElement("option");
      option.value = owner;
      option.textContent = owner;
      ownerFilter.appendChild(option);
    });
  } catch (error) {
    console.error(error);
  }
}

function buildQueryParams() {
  const params = new URLSearchParams();
  if (ownerFilter.value) params.set("owner", ownerFilter.value);
  if (dateFromFilter.value) params.set("date_from", dateFromFilter.value);
  if (dateToFilter.value) params.set("date_to", dateToFilter.value);
  if (statusFilter.value.trim()) params.set("status", statusFilter.value.trim());
  return params.toString();
}

function renderSimpleList(container, rows, formatter, emptyText) {
  if (!rows || rows.length === 0) {
    container.textContent = emptyText;
    return;
  }
  container.textContent = rows.map(formatter).join("\n");
}

function renderTable(container, columns, rows, emptyText) {
  if (!rows || rows.length === 0) {
    container.textContent = emptyText;
    return;
  }

  const head = columns.map((column) => `<th>${column.label}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((column) => `<td>${column.render(row)}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  container.innerHTML = `<table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

async function loadDashboard() {
  try {
    const query = buildQueryParams();
    const [dashboardResponse, prioritiesResponse] = await Promise.all([
      fetch(`/dashboard${query ? `?${query}` : ""}`),
      fetch(`/priorities${query ? `?${query}` : ""}`),
    ]);
    if (dashboardResponse.status === 401 || prioritiesResponse.status === 401) {
      return;
    }
    const data = await dashboardResponse.json();
    const prioritiesData = await prioritiesResponse.json();

    totalInteractions.textContent = String(data.total_interactions);
    activityTypes.textContent = data.by_type.length ? String(data.by_type.length) : "0";
    pendingCount.textContent = String(data.pending_tasks.length);
    staleCount.textContent = String(data.stale_contacts.length);

    renderSimpleList(
      ownerLoadList,
      data.owner_load,
      (row) => `${row.owner_name}: ${row.total_records}`,
      "Sin datos de carga."
    );
    renderSimpleList(
      pendingTasksList,
      data.pending_tasks,
      (row) => `${row.owner_name || "Sin owner"} | ${row.contact_name || "Sin contacto"} | ${row.subject} | ${row.status || "Sin estatus"} | ${row.due_date || "Sin fecha"}`,
      "Sin compromisos para los filtros actuales."
    );
    renderSimpleList(
      staleContactsList,
      data.stale_contacts,
      (row) => `${row.related_name} | ${row.owner_name} | ${row.last_touch}`,
      "No hay clientes rezagados con estos filtros."
    );
    renderTable(
      recentActivityList,
      [
        { label: "Cliente", render: (row) => row.related_name || "Sin nombre" },
        { label: "Tipo", render: (row) => row.source_type || "-" },
        { label: "Fecha", render: (row) => row.interaction_at || "-" },
        { label: "Owner", render: (row) => row.owner_name || "Sin owner" },
      ],
      data.recent_activity,
      "Sin actividad reciente."
    );

    renderTable(
      prioritiesTable,
      [
        { label: "Cliente", render: (row) => row.related_name || "-" },
        { label: "Vendedor", render: (row) => row.owner_name || "-" },
        { label: "Prioridad", render: (row) => `<span class="priority-badge ${row.priority_label}">${row.priority_label}</span>` },
        { label: "Score", render: (row) => String(row.score ?? "-") },
        { label: "Último toque", render: (row) => row.last_touch || "-" },
        { label: "Razones", render: (row) => (row.reasons || []).join("<br>") },
      ],
      prioritiesData.items || [],
      "Sin prioridades calculadas."
    );
  } catch (error) {
    console.error(error);
  }
}

async function loadHistory() {
  try {
    const response = await fetch("/history?limit=12");
    if (response.status === 401) {
      return;
    }
    const data = await response.json();
    if (!data.length) {
      historyList.textContent = "Todavía no hay preguntas registradas.";
      return;
    }

    historyList.innerHTML = "";
    data.forEach((item) => {
      const wrapper = document.createElement("article");
      wrapper.className = "history-item";
      wrapper.innerHTML = `
        <div class="history-meta">${item.created_at} | modo ${item.mode || "-"}</div>
        <h3 class="history-question">${item.question}</h3>
        <p class="history-answer">${item.answer}</p>
      `;
      wrapper.addEventListener("click", () => {
        questionInput.value = item.question;
        answerOutput.textContent = item.answer;
        modePill.textContent = `Modo: ${item.mode || "-"}`;
        sourcePill.textContent = `Fuentes: ${item.sources || "-"}`;
        webPill.textContent = `Web: ${item.used_web ? "sí" : "no"}`;
      });
      historyList.appendChild(wrapper);
    });
  } catch (error) {
    historyList.textContent = "No se pudo cargar el historial.";
  }
}

document.querySelectorAll(".sample-chip").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question || "";
    questionInput.focus();
  });
});

askButton.addEventListener("click", askQuestion);
refreshButton.addEventListener("click", refreshData);
loginButton.addEventListener("click", login);
logoutButton.addEventListener("click", logout);
applyFiltersButton.addEventListener("click", loadDashboard);
clearFiltersButton.addEventListener("click", async () => {
  ownerFilter.value = "";
  dateFromFilter.value = "";
  dateToFilter.value = "";
  statusFilter.value = "";
  await loadDashboard();
});
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askQuestion();
  }
});
loginPassword.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    login();
  }
});

checkHealth();
fetchCurrentUser().then((user) => {
  if (user) {
    loadOwners();
    loadDashboard();
    loadHistory();
  }
});
