/**
 * B08 — AEB upper-control plugin page.
 *
 * Runs inside the Host's restricted plugin iframe.  Everything goes through
 * `window.AstrBotPluginPage`: the page never reads parent cookies or tokens and
 * never talks to ZeroMQ itself.
 *
 * Response contract (C03): the bridge resolves with the business object and
 * rejects with `Error(message)` only.  HTTP status and `data` are dropped, so
 * every rejection carries a stable `[UPPER:<code>]` prefix that is parsed here.
 */

const bridge = window.AstrBotPluginPage;

const ENDPOINT = {
  settings: "upper/settings",
  settingsSave: "upper/settings/save",
  providers: "upper/providers",
  actions: "upper/actions",
  actionsSave: "upper/actions/save",
  actionsDelete: "upper/actions/delete",
  prompts: "upper/prompts",
  promptsSave: "upper/prompts/save",
  plans: "upper/plans",
  planGet: "upper/plans/get",
};

const PROMPT_KINDS = ["routing", "planning"];

const LEVELS = ["primitive", "basic", "skill"];

const PHASE_LABEL = {
  unsent: "未发送",
  sending: "投递中",
  received: "已接收",
  injected: "已注入",
  rejected: "拒绝",
  unknown: "未知",
};

const PHASE_STATE = {
  unsent: "",
  sending: "warn",
  received: "warn",
  injected: "ok",
  rejected: "bad",
  unknown: "warn",
};

const ROUTE_LABEL = {
  speech_only: "只说话",
  action_only: "只执行",
  speech_and_action: "说话并执行",
};

/** Every rejection is shown with the same wording, keyed by the stable code. */
const ERROR_TEXT = {
  revision_conflict: "内容已被其他页面更新。请重新载入后再改，或先复制你的草稿。",
  invalid_request: "提交内容不符合要求，请检查标红的字段。",
  field_not_allowed: "提交里包含不允许的字段，请移除后重试。",
  path_not_allowed: "不接受文件路径。这里只填写动作 ID 或文本内容。",
  cursor_invalid: "分页游标无效，已停在第一页。",
  not_found: "找不到该条目，可能已被删除。",
  dependency_unavailable: "后端依赖暂未就绪。稍后重试，输入不会丢失。",
  internal_error: "服务端处理失败。输入已保留，可稍后重试。",
  unknown: "请求失败。输入已保留，可稍后重试。",
};

const state = {
  context: null,
  tab: "actions",
  settings: null,
  providers: null,
  actions: { records: [], revision: 0, catalogVersion: "", ex: null },
  prompts: null,
  plans: { items: [], nextCursor: null },
  filter: { text: "", level: "all" },
  draft: null,
  draftBase: null,
  selectedPlanId: null,
  busy: { saveAction: false, saveSettings: false, savePrompt: {} },
};

/* ─────────────────────────────── helpers ─────────────────────────────── */

const $ = (id) => document.getElementById(id);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function setText(node, value) {
  if (!node) return;
  node.textContent = value === null || value === undefined ? "" : String(value);
}

function clone(value) {
  return value === null || value === undefined ? value : JSON.parse(JSON.stringify(value));
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function t(key, fallback) {
  if (bridge && typeof bridge.t === "function") {
    try {
      return bridge.t(key, fallback);
    } catch {
      return fallback;
    }
  }
  return fallback;
}

function applyI18n() {
  $$("[data-i18n]").forEach((node) => {
    const key = node.dataset.i18n;
    const fallback = node.textContent.trim();
    setText(node, t(key, fallback));
  });
  document.title = t("pages.upper-control.title", "上位控制");
}

let toastTimer = null;
function toast(message, tone) {
  const node = $("toast");
  setText(node, message);
  node.dataset.tone = tone || "";
  node.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    node.hidden = true;
  }, 3600);
}

/** Banner with up to two recovery actions.  `null` hides a button. */
function showBanner({ text, tone, primary, secondary }) {
  const banner = $("banner");
  setText($("banner-text"), text);
  banner.dataset.tone = tone || "";
  banner.hidden = false;

  const primaryBtn = $("banner-primary");
  const secondaryBtn = $("banner-secondary");
  bindBannerButton(primaryBtn, primary);
  bindBannerButton(secondaryBtn, secondary);
}

function bindBannerButton(button, spec) {
  button.hidden = !spec;
  button.onclick = null;
  if (!spec) return;
  setText(button, spec.label);
  button.onclick = () => {
    hideBanner();
    spec.run();
  };
}

function hideBanner() {
  $("banner").hidden = true;
}

function setFieldError(fieldId, errorId, message) {
  const field = $(fieldId);
  const error = $(errorId);
  if (field) {
    if (message) {
      field.setAttribute("aria-invalid", "true");
    } else {
      field.removeAttribute("aria-invalid");
    }
  }
  if (error) {
    setText(error, message || "");
    error.hidden = !message;
  }
}

function setSaveState(node, message, tone) {
  setText(node, message || "");
  node.dataset.tone = tone || "";
}

/* ─────────────────────────────── api adapter ─────────────────────────── */

function parseRejection(error) {
  const raw = error && typeof error.message === "string" ? error.message : "";
  const match = raw.match(/\[UPPER:([a-z_]+)\]/);
  const code = match ? match[1] : "unknown";
  const detail = match ? raw.slice(match.index + match[0].length).trim() : raw;
  return {
    code,
    message: ERROR_TEXT[code] || ERROR_TEXT.unknown,
    detail,
  };
}

async function apiGet(endpoint, params) {
  try {
    const data = await bridge.apiGet(endpoint, params || {});
    return { ok: true, data };
  } catch (error) {
    return { ok: false, error: parseRejection(error) };
  }
}

async function apiPost(endpoint, body) {
  try {
    const data = await bridge.apiPost(endpoint, body || {});
    return { ok: true, data };
  } catch (error) {
    return { ok: false, error: parseRejection(error) };
  }
}

/** One place that decides how a failed call is surfaced. */
function reportFailure(error, { onConflict, onReload } = {}) {
  if (error.code === "revision_conflict") {
    showBanner({
      text: `${error.message}${error.detail ? `（${error.detail}）` : ""}`,
      tone: "warn",
      primary: onReload ? { label: "重新载入", run: onReload } : null,
      secondary: onConflict ? { label: "复制草稿", run: onConflict } : null,
    });
    return;
  }
  showBanner({
    text: `${error.message}${error.detail ? `（${error.detail}）` : ""}`,
    tone: error.code === "dependency_unavailable" ? "warn" : "",
    primary: { label: "刷新", run: () => refreshAll() },
    secondary: null,
  });
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("草稿已复制到剪贴板。", "ok");
  } catch {
    showBanner({
      text: "浏览器拒绝了剪贴板访问。请手动选中下方内容后复制。",
      tone: "warn",
      primary: null,
      secondary: null,
    });
    $("banner-text").appendChild(document.createElement("br"));
    const pre = document.createElement("pre");
    pre.className = "preview";
    pre.textContent = text;
    $("banner-text").appendChild(pre);
  }
}

/* ─────────────────────────────── confirm dialog ──────────────────────── */

let dialogResolve = null;

function askConfirm({ title, body, okText, danger = false }) {
  const dialog = $("confirm-dialog");
  if (dialog.open) dialog.close("cancel");
  setText($("confirm-title"), title);
  setText($("confirm-body"), body);
  setText($("confirm-ok"), okText || "继续");
  $("confirm-ok").className = danger ? "btn btn-danger" : "btn btn-primary";
  $("confirm-cancel").textContent = "取消";

  return new Promise((resolve) => {
    dialogResolve = resolve;
    dialog.showModal();
  });
}

function wireConfirmDialog() {
  const dialog = $("confirm-dialog");
  dialog.addEventListener("close", () => {
    const resolve = dialogResolve;
    dialogResolve = null;
    if (resolve) resolve(dialog.returnValue === "ok");
  });
}

/* ─────────────────────────────── status chips ────────────────────────── */

function renderStatus() {
  const runtime = state.settings && state.settings.runtime ? state.settings.runtime : null;
  const controlChip = $("status-control");
  const controlText = $("status-control-text");

  if (!runtime || runtime.control_state === "unavailable") {
    controlChip.dataset.state = "unknown";
    setText(controlText, "通道未装配");
  } else if (runtime.control_state === "running") {
    controlChip.dataset.state = "ok";
    setText(
      controlText,
      runtime.input_ready ? "通道运行中 · 可接收" : "通道运行中 · 输入未就绪",
    );
  } else if (runtime.control_state === "stopped") {
    controlChip.dataset.state = "warn";
    setText(controlText, "通道已停止");
  } else {
    controlChip.dataset.state = "bad";
    setText(controlText, "通道异常");
  }

  const exChip = $("status-ex");
  const ex = state.actions.ex;
  if (!ex || ex.state === "unknown") {
    exChip.dataset.state = "unknown";
    setText($("status-ex-text"), "EX 未知");
  } else if (ex.state === "online") {
    exChip.dataset.state = "ok";
    setText($("status-ex-text"), `EX 在线 · ${(ex.action_ids || []).length} 个动作`);
  } else {
    exChip.dataset.state = "warn";
    setText($("status-ex-text"), "EX 离线");
  }

  setText(
    $("status-revision"),
    state.settings ? `rev ${state.settings.revision}` : "rev —",
  );
}

/* ─────────────────────────────── actions ─────────────────────────────── */

function emptyDraft() {
  return {
    action_ref: null,
    action_id: null,
    level: "primitive",
    name: "",
    description: "",
    parameters: [],
    usage_stages: [],
    skill_markdown: "",
    enabled: true,
  };
}

function visibleRecords() {
  const text = state.filter.text.trim().toLowerCase();
  return state.actions.records.filter((record) => {
    if (state.filter.level !== "all" && record.level !== state.filter.level) return false;
    if (!text) return true;
    return [record.name, record.description, record.action_id, record.action_ref]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(text));
  });
}

function renderActionList() {
  const list = $("action-list");
  const rows = visibleRecords();
  list.textContent = "";
  const template = $("action-item-template");

  rows.forEach((record) => {
    const item = template.content.firstElementChild.cloneNode(true);
    const button = item.querySelector(".action-item-button");
    item.dataset.enabled = String(record.enabled !== false);
    setText(item.querySelector(".action-item-name"), record.name || "(未命名)");
    const meta = record.action_id ? `${record.level} · ${record.action_id}` : record.level;
    setText(item.querySelector(".action-item-meta"), meta);
    const active =
      state.draft &&
      ((state.draft.action_ref && state.draft.action_ref === record.action_ref) ||
        (!state.draft.action_ref && state.draft.action_id === record.action_id));
    if (active) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => selectRecord(record.action_ref));
    list.appendChild(item);
  });

  $("action-empty").hidden = rows.length > 0;
  if (!rows.length) {
    setText(
      $("action-empty"),
      state.actions.records.length ? "没有匹配的条目。" : "还没有动作条目。",
    );
  }
}

function actionDirty() {
  if (!state.draft || !state.draftBase) return false;
  return !sameJson(state.draft, state.draftBase);
}

async function guardDraft() {
  if (!actionDirty()) return true;
  const proceed = await askConfirm({
    title: "有未保存的改动",
    body: "当前条目的改动还没有保存。离开会丢弃这些改动。",
    okText: "丢弃并离开",
    danger: true,
  });
  if (proceed) {
    state.draft = null;
    state.draftBase = null;
    renderEditor();
  }
  return proceed;
}

async function selectRecord(actionRef) {
  if (!(await guardDraft())) return;
  const record = state.actions.records.find((item) => item.action_ref === actionRef);
  if (!record) return;
  state.draft = clone(record);
  state.draftBase = clone(record);
  renderActionList();
  renderEditor();
}

async function startNewAction() {
  if (!(await guardDraft())) return;
  state.draft = emptyDraft();
  state.draftBase = emptyDraft();
  renderActionList();
  renderEditor();
  $("field-name").focus();
}

function renderEditor() {
  const form = $("action-form");
  const empty = $("editor-empty");
  if (!state.draft) {
    form.hidden = true;
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  form.hidden = false;

  const draft = state.draft;
  setText($("editor-title"), draft.action_ref ? "动作详情" : "新增动作");
  setText($("action-ref-badge"), draft.action_ref || "未保存");

  const ex = state.actions.ex;
  const badge = $("action-ex-badge");
  if (!ex || ex.state !== "online") {
    badge.dataset.state = "unknown";
    setText(badge, "EX 离线 · 仅编辑");
  } else if (draft.action_id && (ex.action_ids || []).includes(draft.action_id)) {
    badge.dataset.state = "ok";
    setText(badge, "EX 已注册");
  } else if (draft.action_id) {
    badge.dataset.state = "warn";
    setText(badge, "EX 未注册 · 仅描述");
  } else {
    badge.dataset.state = "";
    setText(badge, "独立 Skill · 不关联可执行 ID");
  }

  $("field-name").value = draft.name || "";
  $("field-level").value = draft.level || "primitive";
  $("field-action-id").value = draft.action_id || "";
  $("field-description").value = draft.description || "";
  $("field-stages").value = (draft.usage_stages || []).join(", ");
  $("field-skill").value = draft.skill_markdown || "";
  $("field-enabled").checked = draft.enabled !== false;

  syncLevelUi();
  renderParamRows();
  hideSkillPreview();
  setSaveState($("save-state"), "");
  clearActionErrors();
}

function syncLevelUi() {
  const level = $("field-level").value;
  const wrap = $("field-action-id-wrap");
  const input = $("field-action-id");
  const isSkill = level === "skill";
  wrap.hidden = false;
  input.disabled = isSkill;
  if (isSkill) {
    input.value = "";
    setText(
      $("hint-action-id"),
      "skill 层是独立说明，不关联可执行 ID，也不会出现在 EX 动作列表里。",
    );
  } else {
    setText($("hint-action-id"), "只填写 EX 已注册的动作 ID。此处不提交执行权限或机器契约。");
  }
  setText(
    $("readonly-note"),
    isSkill
      ? "skill 层的可用性由是否启用决定；EX 动作能力仍然只读呈现。"
      : "运行可用性来自 EX 能力快照，不能在这里勾选制造。",
  );
}

function renderParamRows() {
  const body = $("param-rows");
  body.textContent = "";
  const params = state.draft && Array.isArray(state.draft.parameters) ? state.draft.parameters : [];
  if (!params.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.className = "hint";
    setText(cell, "还没有参数说明。");
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  params.forEach((param, index) => body.appendChild(buildParamRow(param, index)));
}

function buildParamRow(param, index) {
  const row = $("param-row-template").content.firstElementChild.cloneNode(true);
  row.dataset.index = String(index);
  row.querySelector('[data-role="param-name"]').value = param.name || "";
  row.querySelector('[data-role="param-description"]').value = param.description || "";
  row.querySelector('[data-role="param-unit"]').value = param.unit || "";
  row.querySelector('[data-role="param-required"]').checked = param.required === true;

  const labelIndex = ` ${index + 1}`;
  row.querySelectorAll(".sr-only").forEach((label) => {
    setText(label, label.textContent + labelIndex);
  });
  row
    .querySelector('[data-role="param-remove"]')
    .addEventListener("click", () => removeParam(index));
  return row;
}

function collectParams() {
  const rows = $$("#param-rows .param-row");
  return rows.map((row) => ({
    name: row.querySelector('[data-role="param-name"]').value.trim(),
    description: row.querySelector('[data-role="param-description"]').value.trim(),
    unit: row.querySelector('[data-role="param-unit"]').value.trim(),
    required: row.querySelector('[data-role="param-required"]').checked,
  }));
}

function syncDraftFromForm() {
  if (!state.draft) return;
  const draft = state.draft;
  draft.name = $("field-name").value.trim();
  draft.level = $("field-level").value;
  draft.action_id = draft.level === "skill" ? null : $("field-action-id").value.trim() || null;
  draft.description = $("field-description").value;
  draft.usage_stages = $("field-stages")
    .value.split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  draft.skill_markdown = $("field-skill").value;
  draft.enabled = $("field-enabled").checked;
  draft.parameters = collectParams().filter((param) => param.name || param.description);
}

function clearActionErrors() {
  ["name", "level", "action-id", "description"].forEach((name) => {
    setFieldError(`field-${name}`, `error-${name}`, "");
  });
}

function validateDraft() {
  clearActionErrors();
  const draft = state.draft;
  let firstInvalid = null;
  const fail = (field, errorField, message) => {
    setFieldError(field, errorField, message);
    if (!firstInvalid) firstInvalid = $(field);
  };

  if (!draft.name) {
    fail("field-name", "error-name", "请填写名称，便于在目录里识别这个动作。");
  }
  if (draft.level !== "skill" && !draft.action_id) {
    fail(
      "field-action-id",
      "error-action-id",
      "primitive / basic 需要填写 EX 已注册的 action_id。",
    );
  }
  if (draft.action_id && /^[A-Za-z]:[\\/]|^[\\/]/.test(draft.action_id)) {
    fail("field-action-id", "error-action-id", "这里只接受动作 ID，不接受文件路径。");
  }
  if (firstInvalid) firstInvalid.focus();
  return !firstInvalid;
}

async function saveAction() {
  if (state.busy.saveAction) return;
  syncDraftFromForm();
  if (!validateDraft()) {
    setSaveState($("save-state"), "请先修正标红的字段。", "bad");
    return;
  }

  state.busy.saveAction = true;
  const button = $("btn-save-action");
  button.disabled = true;
  setText(button, "保存中…");
  setSaveState($("save-state"), "");

  const record = clone(state.draft);
  const result = await apiPost(ENDPOINT.actionsSave, {
    record,
    expected_revision: state.actions.revision,
  });

  state.busy.saveAction = false;
  button.disabled = false;
  setText(button, "保存");

  if (!result.ok) {
    setSaveState($("save-state"), "保存失败，草稿已保留。", "bad");
    reportFailure(result.error, {
      onConflict: () => copyToClipboard(JSON.stringify(state.draft, null, 2)),
      onReload: async () => {
        state.draft = null;
        state.draftBase = null;
        await loadActions();
        renderEditor();
      },
    });
    return;
  }

  state.actions.revision = result.data.revision;
  const actionRef = result.data.action_ref;
  await loadActions();
  const saved = state.actions.records.find((item) => item.action_ref === actionRef);
  if (saved) {
    state.draft = clone(saved);
    state.draftBase = clone(saved);
  } else {
    state.draft = null;
    state.draftBase = null;
  }
  renderActionList();
  renderEditor();
  setSaveState($("save-state"), `已保存 · rev ${result.data.revision}`, "ok");
  renderStatus();
  toast("动作已保存。", "ok");
}

async function deleteAction() {
  if (!state.draft || !state.draft.action_ref) {
    toast("这个条目还没有保存，无需删除。");
    return;
  }
  const proceed = await askConfirm({
    title: `删除「${state.draft.name || state.draft.action_ref}」？`,
    body: "只删除 AEB 本地目录条目，不会卸载或停用 EX 里的动作。删除后不能恢复。",
    okText: "删除条目",
    danger: true,
  });
  if (!proceed) return;

  const result = await apiPost(ENDPOINT.actionsDelete, {
    action_ref: state.draft.action_ref,
    expected_revision: state.actions.revision,
  });

  if (!result.ok) {
    reportFailure(result.error, {
      onReload: async () => {
        state.draft = null;
        state.draftBase = null;
        await loadActions();
        renderEditor();
      },
    });
    return;
  }

  state.actions.revision = result.data.revision;
  state.draft = null;
  state.draftBase = null;
  await loadActions();
  renderEditor();
  renderStatus();
  toast("条目已删除，EX 动作未被改动。", "ok");
}

function hideSkillPreview() {
  $("preview-skill").hidden = true;
  $("btn-preview-skill").setAttribute("aria-expanded", "false");
}

function toggleSkillPreview() {
  const preview = $("preview-skill");
  const button = $("btn-preview-skill");
  const next = preview.hidden;
  preview.hidden = !next;
  button.setAttribute("aria-expanded", String(next));
  if (next) {
    const text = $("field-skill").value;
    preview.textContent = text || "（空）";
  }
}

/* ─────────────────────────────── prompts ─────────────────────────────── */

function promptDirty(kind) {
  if (!state.prompts) return false;
  const loaded = state.prompts.prompts[kind];
  if (!loaded) return false;
  return ($(`prompt-${kind}`).value || "") !== (loaded.content || "");
}

function anyPromptDirty() {
  return PROMPT_KINDS.some((kind) => promptDirty(kind));
}

function renderPrompts() {
  if (!state.prompts) return;
  PROMPT_KINDS.forEach((kind) => {
    const info = state.prompts.prompts[kind];
    const card = document.querySelector(`.card[data-kind="${kind}"]`);
    if (!card || !info) return;
    $(`prompt-${kind}`).value = info.content || "";

    const badge = card.querySelector('[data-role="source"]');
    const isOverride = info.source === "override";
    badge.dataset.state = isOverride ? "warn" : "";
    setText(badge, isOverride ? "覆盖" : "内置");

    const bytes = info.bytes || 0;
    setText(
      card.querySelector('[data-role="meta"]'),
      `${isOverride ? "当前使用覆盖内容" : "当前使用内置文本"} · ${bytes} 字节 · hash ${String(
        info.hash || "",
      ).slice(0, 12)}`,
    );
  });
  setText(
    $("status-revision"),
    state.settings ? `rev ${state.settings.revision}` : "rev —",
  );
}

function togglePromptPreview(kind) {
  const preview = $(`preview-${kind}`);
  const button = document.querySelector(`[data-role="preview"][data-kind="${kind}"]`);
  const next = preview.hidden;
  preview.hidden = !next;
  button.setAttribute("aria-expanded", String(next));
  if (!next) return;
  const info = state.prompts && state.prompts.prompts[kind];
  const local = $(`prompt-${kind}`).value;
  const effective = local || (info ? info.effective_content : "");
  preview.textContent = effective || "（空）";
}

async function savePrompt(kind) {
  if (state.busy.savePrompt[kind]) return;
  const button = document.querySelector(`[data-role="save"][data-kind="${kind}"]`);
  state.busy.savePrompt[kind] = true;
  button.disabled = true;
  const original = button.textContent;
  setText(button, "保存中…");

  const result = await apiPost(ENDPOINT.promptsSave, {
    kind,
    content: $(`prompt-${kind}`).value,
    expected_revision: state.settings ? state.settings.revision : 0,
  });

  state.busy.savePrompt[kind] = false;
  button.disabled = false;
  setText(button, original);

  if (!result.ok) {
    reportFailure(result.error, {
      onReload: () => refreshAll(),
      onConflict: () => copyToClipboard($(`prompt-${kind}`).value),
    });
    return;
  }

  if (state.settings) state.settings.revision = result.data.revision;
  await Promise.all([loadPrompts(), loadSettings({ quiet: true })]);
  renderStatus();
  toast(kind === "routing" ? "路由规范已保存。" : "编排规范已保存。", "ok");
}

async function restorePrompt(kind) {
  const proceed = await askConfirm({
    title: "恢复内置文本？",
    body: "会以空覆盖保存，模型随后读取插件内置的规范文本。当前的覆盖内容会被清空。",
    okText: "恢复默认",
    danger: true,
  });
  if (!proceed) return;
  $(`prompt-${kind}`).value = "";
  await savePrompt(kind);
}

/* ─────────────────────────────── settings ────────────────────────────── */

function providerOptions(selectedId) {
  const providers = (state.providers && state.providers.providers) || [];
  const known = providers.map((item) => item.id);
  const options = [{ value: "", label: "使用 Host 默认" }];
  providers.forEach((item) => {
    options.push({
      value: item.id,
      label: item.model ? `${item.name} · ${item.model}` : item.name || item.id,
    });
  });
  if (selectedId && !known.includes(selectedId)) {
    options.push({ value: selectedId, label: `${selectedId}（本机不存在）` });
  }
  return options;
}

function fillSelect(node, options, selectedId) {
  node.textContent = "";
  options.forEach((option) => {
    const item = document.createElement("option");
    item.value = option.value;
    setText(item, option.label);
    node.appendChild(item);
  });
  node.value = selectedId || "";
  if (node.value !== (selectedId || "")) node.value = "";
}

function renderSettings() {
  if (!state.settings) return;
  const values = state.settings.settings;
  const providers = state.providers || { providers: [], selected: {}, stale: [] };

  fillSelect(
    $("field-routing-provider"),
    providerOptions(values.routing_provider_id),
    values.routing_provider_id || "",
  );
  fillSelect(
    $("field-generation-provider"),
    providerOptions(values.generation_provider_id),
    values.generation_provider_id || "",
  );

  $("field-timeout").value = values.model_timeout_s;
  $("field-upper-enabled").checked = values.upper_enabled === true;
  $("field-control-enabled").checked = values.control_enabled === true;
  $("field-bind-host").value = values.control_bind_host || "";
  $("field-port").value = values.control_port;
  $("field-peer").value = values.control_peer_id || "";
  $("field-robot").value = values.robot_id || "";

  const stale = providers.stale || [];
  const staleNote = $("provider-stale");
  staleNote.hidden = stale.length === 0;
  setText(
    staleNote,
    stale.length
      ? `这些 Provider 在本机已不存在：${stale.join("、")}。请重新选择，否则模型调用会失败。`
      : "",
  );

  const runtime = state.settings.runtime || {};
  setText(
    $("runtime-note"),
    runtime.control_state === "running"
      ? `实际运行：监听 ${values.control_bind_host}:${values.control_port} · peer ${
          runtime.control_peer || "未设置"
        } · ${runtime.input_ready ? "可接收输入" : "输入未就绪"}`
      : "实际运行：control 通道尚未运行。保存不会启动通道，需要重载插件。",
  );
  $("reload-badge").hidden = true;
  setSaveState($("settings-save-state"), "");
}

function collectSettings() {
  return {
    upper_enabled: $("field-upper-enabled").checked,
    control_enabled: $("field-control-enabled").checked,
    routing_provider_id: $("field-routing-provider").value,
    generation_provider_id: $("field-generation-provider").value,
    control_bind_host: $("field-bind-host").value.trim(),
    control_port: Number($("field-port").value),
    control_peer_id: $("field-peer").value.trim(),
    robot_id: $("field-robot").value.trim(),
    model_timeout_s: Number($("field-timeout").value),
  };
}

function settingsDirty() {
  if (!state.settings) return false;
  const values = state.settings.settings;
  const current = collectSettings();
  return Object.keys(current).some((key) => {
    const before = values[key];
    const after = current[key];
    if (typeof before === "number" || typeof after === "number") {
      return Number(before) !== Number(after);
    }
    return before !== after;
  });
}

function validateSettings() {
  let ok = true;
  const port = Number($("field-port").value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    setFieldError("field-port", "error-port", "端口需要是 1 到 65535 之间的整数。");
    ok = false;
  } else {
    setFieldError("field-port", "error-port", "");
  }
  const timeout = Number($("field-timeout").value);
  if (!Number.isFinite(timeout) || timeout < 1 || timeout > 600) {
    setFieldError("field-timeout", "error-timeout", "超时需要是 1 到 600 秒之间的数值。");
    ok = false;
  } else {
    setFieldError("field-timeout", "error-timeout", "");
  }
  const host = $("field-bind-host").value.trim();
  if (!host || host.includes("/") || host.includes("://")) {
    setFieldError(
      "field-bind-host",
      "error-bind-host",
      "这里只填写主机名或 IP，例如 127.0.0.1。",
    );
    ok = false;
  } else {
    setFieldError("field-bind-host", "error-bind-host", "");
  }
  if (!ok) {
    const invalid = document.querySelector("#settings-form [aria-invalid='true']");
    if (invalid) invalid.focus();
  }
  return ok;
}

async function saveSettings() {
  if (state.busy.saveSettings) return;
  if (!state.settings) return;
  if (!validateSettings()) {
    setSaveState($("settings-save-state"), "请先修正标红的字段。", "bad");
    return;
  }

  const current = collectSettings();
  const before = state.settings.settings;
  const changed = {};
  Object.keys(current).forEach((key) => {
    const differs =
      typeof current[key] === "number"
        ? Number(before[key]) !== Number(current[key])
        : before[key] !== current[key];
    if (differs) changed[key] = current[key];
  });

  if (!Object.keys(changed).length) {
    setSaveState($("settings-save-state"), "没有需要保存的修改。");
    return;
  }

  state.busy.saveSettings = true;
  const button = $("btn-save-settings");
  button.disabled = true;
  setText(button, "保存中…");

  const result = await apiPost(ENDPOINT.settingsSave, {
    settings: changed,
    expected_revision: state.settings.revision,
  });

  state.busy.saveSettings = false;
  button.disabled = false;
  setText(button, "保存设置");

  if (!result.ok) {
    setSaveState($("settings-save-state"), "保存失败，输入已保留。", "bad");
    reportFailure(result.error, { onReload: () => refreshAll() });
    return;
  }

  state.settings.revision = result.data.revision;
  await loadSettings({ quiet: true });
  renderStatus();

  const reload = result.data.reload_required || [];
  $("reload-badge").hidden = reload.length === 0;
  if (reload.length) {
    setSaveState(
      $("settings-save-state"),
      `已保存。以下字段需要重载插件后生效：${reload.join("、")}`,
      "ok",
    );
  } else {
    setSaveState($("settings-save-state"), "已保存，下次输入即生效。", "ok");
  }
  toast("设置已保存。保存不会向 EX 发送测试任务。", "ok");
}

/* ─────────────────────────────── plans ───────────────────────────────── */

function phaseBadge(node, phase) {
  const normalised = PHASE_LABEL[phase] ? phase : "unknown";
  node.dataset.state = PHASE_STATE[normalised] || "";
  setText(node, PHASE_LABEL[normalised]);
}

function renderPlanRows() {
  const body = $("record-rows");
  body.textContent = "";
  const template = $("record-row-template");

  state.plans.items.forEach((plan) => {
    const row = template.content.firstElementChild.cloneNode(true);
    setText(row.querySelector(".cell-plan"), plan.plan_id);
    setText(row.querySelector(".cell-route"), ROUTE_LABEL[plan.route] || plan.route || "—");
    setText(row.querySelector(".cell-total"), plan.segment_total);
    setText(row.querySelector(".cell-sent"), plan.sent_prefix);
    setText(row.querySelector(".cell-pending"), plan.pending_suffix);
    phaseBadge(row.querySelector(".cell-phase"), plan.delivery_phase);
    setText(
      row.querySelector(".cell-version"),
      `plan r${plan.plan_revision ?? "—"} / ${String(plan.catalog_version || "—").slice(0, 8)}`,
    );
    if (plan.plan_id === state.selectedPlanId) row.setAttribute("aria-current", "true");
    row.addEventListener("click", () => loadPlanDetail(plan.plan_id));
    body.appendChild(row);
  });

  $("record-empty").hidden = state.plans.items.length > 0;
  $("btn-more-plans").hidden = !state.plans.nextCursor;
}

function renderPlanDetail(plan) {
  const card = $("record-detail");
  if (!plan) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  setText($("record-detail-title"), plan.plan_id);
  phaseBadge($("record-detail-phase"), plan.delivery_phase);

  const meta = $("record-detail-meta");
  meta.textContent = "";
  const rows = [
    ["路由", ROUTE_LABEL[plan.route] || plan.route || "—"],
    ["段落", `${plan.segment_total} 段 · 已发前缀 ${plan.sent_prefix} · 待发后缀 ${plan.pending_suffix}`],
    ["计划版本", `r${plan.plan_revision ?? "—"}`],
    ["提示词版本", String(plan.prompt_version || "—").slice(0, 16)],
    ["目录版本", String(plan.catalog_version || "—").slice(0, 16)],
    ["创建时间", plan.created_at || "—"],
  ];
  rows.forEach(([label, value]) => {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    setText(dt, label);
    setText(dd, value);
    wrap.append(dt, dd);
    meta.appendChild(wrap);
  });

  const segmentList = $("record-segments");
  segmentList.textContent = "";
  (plan.segments || []).forEach((segment, index) => {
    const item = document.createElement("li");
    item.dataset.pending = String(index >= plan.sent_prefix);
    const metaLine = document.createElement("span");
    metaLine.className = "segment-meta";
    setText(
      metaLine,
      `${segment.segment_id || `#${segment.ordinal}`} · ${
        index < plan.sent_prefix ? "本次已发送" : "留在 AEB"
      }`,
    );
    item.appendChild(metaLine);
    item.appendChild(document.createTextNode(segment.text || ""));
    segmentList.appendChild(item);
  });

  const deliveryList = $("record-deliveries");
  deliveryList.textContent = "";
  const deliveries = plan.deliveries || [];
  if (!deliveries.length) {
    const item = document.createElement("li");
    setText(item, "还没有投递记录。");
    deliveryList.appendChild(item);
  }
  deliveries.forEach((delivery) => {
    const item = document.createElement("li");
    const badge = document.createElement("span");
    badge.className = "badge";
    phaseBadge(badge, delivery.phase);
    item.appendChild(badge);
    const detail = document.createElement("span");
    setText(
      detail,
      ` · ${delivery.batch_id || "—"} · ${String(delivery.payload_hash || "—").slice(0, 12)} · ${
        delivery.attempted_at || "—"
      }`,
    );
    item.appendChild(detail);
    deliveryList.appendChild(item);
  });
}

async function loadPlanDetail(planId) {
  const result = await apiGet(ENDPOINT.planGet, { plan_id: planId });
  if (!result.ok) {
    reportFailure(result.error, { onReload: () => refreshAll() });
    return;
  }
  state.selectedPlanId = planId;
  renderPlanRows();
  renderPlanDetail(result.data.plan);
}

/* ─────────────────────────────── loaders ─────────────────────────────── */

async function loadSettings({ quiet = false } = {}) {
  const result = await apiGet(ENDPOINT.settings);
  if (!result.ok) {
    state.settings = null;
    renderStatus();
    if (!quiet) {
      showBanner({
        text: `${result.error.message} 设置与提示词暂时不可编辑。`,
        tone: "warn",
        primary: { label: "重试", run: () => refreshAll() },
        secondary: null,
      });
    }
    return result;
  }
  state.settings = result.data;
  renderSettings();
  renderStatus();
  return result;
}

async function loadProviders() {
  const result = await apiGet(ENDPOINT.providers);
  if (result.ok) {
    state.providers = result.data;
    if (state.settings) renderSettings();
  }
  return result;
}

async function loadActions() {
  const result = await apiGet(ENDPOINT.actions);
  if (!result.ok) {
    showBanner({
      text: result.error.message,
      tone: "warn",
      primary: { label: "重试", run: () => refreshAll() },
      secondary: null,
    });
    return result;
  }
  state.actions.records = result.data.actions || [];
  state.actions.revision = result.data.revision;
  state.actions.catalogVersion = result.data.catalog_version || "";
  state.actions.ex = result.data.ex_availability || null;
  renderActionList();
  renderStatus();
  return result;
}

async function loadPrompts() {
  const result = await apiGet(ENDPOINT.prompts);
  if (!result.ok) return result;
  state.prompts = result.data;
  renderPrompts();
  return result;
}

async function loadPlans({ append = false } = {}) {
  const params = { limit: 20 };
  if (append && state.plans.nextCursor) params.cursor = state.plans.nextCursor;
  const result = await apiGet(ENDPOINT.plans, params);
  if (!result.ok) {
    reportFailure(result.error, { onReload: () => refreshAll() });
    return result;
  }
  const incoming = result.data.plans || [];
  state.plans.items = append ? state.plans.items.concat(incoming) : incoming;
  state.plans.nextCursor = result.data.next_cursor || null;
  renderPlanRows();
  if (!append) {
    state.selectedPlanId = null;
    renderPlanDetail(null);
  }
  return result;
}

async function refreshAll() {
  hideBanner();
  const button = $("btn-refresh");
  button.disabled = true;
  setText(button, "刷新中…");
  // Initial reads complete independently. Editing before all five reads finish
  // can let a late prompt/provider response replace a new draft.
  const main = $("main");
  main.inert = true;
  main.setAttribute("aria-busy", "true");
  try {
    await Promise.all([loadSettings(), loadProviders(), loadActions(), loadPrompts(), loadPlans()]);
  } finally {
    main.inert = false;
    main.removeAttribute("aria-busy");
    button.disabled = false;
    setText(button, t("pages.upper-control.refresh", "刷新"));
  }
}

/* ─────────────────────────────── tabs ────────────────────────────────── */

function anyDirty(tab) {
  if (tab === "actions") return actionDirty();
  if (tab === "prompts") return anyPromptDirty();
  if (tab === "settings") return settingsDirty();
  return false;
}

async function switchTab(next) {
  if (next === state.tab) return;
  if (anyDirty(state.tab)) {
    const proceed = await askConfirm({
      title: "有未保存的改动",
      body: "切换分区会丢弃当前未保存的改动。",
      okText: "丢弃并切换",
      danger: true,
    });
    if (!proceed) return;
  }

  state.tab = next;
  $$(".tab").forEach((tab) => {
    const selected = tab.dataset.panel === next;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  $$(".panel").forEach((panel) => {
    panel.hidden = panel.id !== `panel-${next}`;
  });
  if (next === "records") await loadPlans();
}

function wireTabs() {
  const tabs = $$(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.panel));
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      let target = null;
      if (event.key === "ArrowRight") target = tabs[(index + 1) % tabs.length];
      if (event.key === "ArrowLeft") target = tabs[(index - 1 + tabs.length) % tabs.length];
      if (event.key === "Home") target = tabs[0];
      if (event.key === "End") target = tabs[tabs.length - 1];
      if (!target) return;
      event.preventDefault();
      target.focus();
      switchTab(target.dataset.panel);
    });
  });
}

/* ─────────────────────────────── wiring ──────────────────────────────── */

function wireActions() {
  $("btn-refresh").addEventListener("click", () => refreshAll());
  $("banner-dismiss").addEventListener("click", () => hideBanner());

  $("action-search").addEventListener("input", (event) => {
    state.filter.text = event.target.value;
    renderActionList();
  });

  $$(".chip-toggle").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.filter.level = chip.dataset.level;
      $$(".chip-toggle").forEach((other) => {
        other.setAttribute("aria-pressed", String(other === chip));
      });
      renderActionList();
    });
  });

  $("btn-new-action").addEventListener("click", () => startNewAction());
  $("btn-save-action").addEventListener("click", () => saveAction());
  $("btn-delete-action").addEventListener("click", () => deleteAction());
  $("btn-discard-action").addEventListener("click", async () => {
    if (!state.draft) return;
    if (state.draft.action_ref) {
      await selectRecord(state.draft.action_ref);
      toast("已恢复为上次保存的内容。");
      return;
    }
    if (!actionDirty() || (await askConfirm({
      title: "丢弃新草稿？",
      body: "这个草稿还没有保存，丢弃后需要重新填写。",
      okText: "丢弃草稿",
      danger: true,
    }))) {
      state.draft = null;
      state.draftBase = null;
      renderActionList();
      renderEditor();
    }
  });

  $("field-level").addEventListener("change", () => {
    syncDraftFromForm();
    syncLevelUi();
  });

  // Every edit must reach the draft, otherwise the unsaved-changes guard and
  // the revision CAS would both work off a stale copy.
  const onActionInput = (event) => {
    syncDraftFromForm();
    if (event && event.target) {
      const id = event.target.id;
      if (id === "field-action-id") setFieldError("field-action-id", "error-action-id", "");
      if (id === "field-name") setFieldError("field-name", "error-name", "");
    }
  };

  ["field-name", "field-action-id", "field-description", "field-stages", "field-skill"].forEach(
    (id) => {
      $(id).addEventListener("input", onActionInput);
    },
  );
  $("field-enabled").addEventListener("change", onActionInput);

  $("param-rows").addEventListener("input", onActionInput);
  $("param-rows").addEventListener("change", onActionInput);

  $("btn-add-param").addEventListener("click", () => {
    syncDraftFromForm();
    state.draft.parameters = (state.draft.parameters || []).concat([
      { name: "", description: "", unit: "", required: false },
    ]);
    renderParamRows();
    const rows = $$("#param-rows .param-row");
    const last = rows[rows.length - 1];
    if (last) last.querySelector('[data-role="param-name"]').focus();
  });

  $("btn-preview-skill").addEventListener("click", () => toggleSkillPreview());

  PROMPT_KINDS.forEach((kind) => {
    document
      .querySelector(`[data-role="save"][data-kind="${kind}"]`)
      .addEventListener("click", () => savePrompt(kind));
    document
      .querySelector(`[data-role="restore"][data-kind="${kind}"]`)
      .addEventListener("click", () => restorePrompt(kind));
    document
      .querySelector(`[data-role="preview"][data-kind="${kind}"]`)
      .addEventListener("click", () => togglePromptPreview(kind));
  });

  $("btn-save-settings").addEventListener("click", () => saveSettings());
  $("btn-more-plans").addEventListener("click", () => loadPlans({ append: true }));
}

function removeParam(index) {
  syncDraftFromForm();
  state.draft.parameters = (state.draft.parameters || []).filter((_item, i) => i !== index);
  renderParamRows();
}

function wireUnloadGuard() {
  window.addEventListener("beforeunload", (event) => {
    if (anyDirty(state.tab)) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  const off = bridge && typeof bridge.onContext === "function"
    ? bridge.onContext(() => applyI18n())
    : null;
  if (off) window.addEventListener("beforeunload", off);
}

/* ─────────────────────────────── boot ────────────────────────────────── */

function fatal(message) {
  showBanner({
    text: message,
    tone: "",
    primary: { label: "重试", run: () => window.location.reload() },
    secondary: null,
  });
}

async function boot() {
  wireConfirmDialog();

  if (!bridge || typeof bridge.ready !== "function") {
    fatal("没有检测到插件页面桥接。请从插件详情页重新打开这个页面。");
    return;
  }

  try {
    state.context = await bridge.ready();
  } catch {
    fatal("插件页面桥接没有就绪。请刷新页面或重新打开插件详情。");
    return;
  }

  applyI18n();
  wireTabs();
  wireActions();
  wireUnloadGuard();
  renderEditor();
  renderPlanRows();
  await refreshAll();
}

boot();
