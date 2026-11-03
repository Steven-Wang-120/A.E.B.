/**
 * B08 — Playwright spec for the AEB upper-control plugin page.
 *
 * Two modes, selected by environment:
 *
 *   fake_bridge (default)  Serves `pages/upper-control/` from a local static
 *                          server and injects a bridge double that talks to a
 *                          Node-side fixture backend.  Asserts requests, save
 *                          results and DOM state — not screenshots.
 *
 *   real_host              Uses a running Dashboard.  Requires
 *                          UPPER_HOST_URL and either UPPER_HOST_STORAGE (a
 *                          Playwright storageState JSON) or UPPER_HOST_COOKIE.
 *
 * When Chromium, the Playwright module or the Host is unavailable the suite
 * reports an explicit `NOT_RUN: ...` skip and writes `upper_control.NOT_RUN.txt`
 * next to this file.  It never claims a pass it did not observe.
 *
 * Run: node --test tests/ui/upper_control.spec.mjs
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { after, before, test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PAGE_ROOT = path.resolve(HERE, "../../pages/upper-control");
const ARTIFACT_DIR = process.env.UPPER_UI_EVIDENCE_DIR || path.join(HERE, "artifacts");
const NOT_RUN_FILE = path.join(ARTIFACT_DIR, "upper_control.NOT_RUN.txt");
const EVIDENCE_FILE = path.join(ARTIFACT_DIR, "upper_control_evidence.json");
const PLUGIN_NAME = "astrbot_plugin_astrbotex_interaction";

const MODE = (process.env.UPPER_UI_MODE || "fake_bridge").toLowerCase();
const VIEWPORT_WIDE = { width: 1280, height: 900 };
const VIEWPORT_NARROW = { width: 390, height: 844 };

const runtime = {
  playwright: null,
  browser: null,
  server: null,
  baseUrl: "",
  reason: "",
  extra: {},
};

/* ─────────────────────────── runtime discovery ─────────────────────────── */

async function loadPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_MODULE,
    "playwright",
    "playwright-core",
    "@playwright/test",
  ].filter(Boolean);
  for (const candidate of candidates) {
    const specifier = normalizeModuleSpecifier(candidate);
    try {
      const loaded = await import(specifier);
      const module = loaded && loaded.chromium ? loaded : loaded.default;
      if (module && module.chromium) return module;
    } catch {
      /* try the next candidate */
    }
  }
  return null;
}

/** Accept a bare specifier, an absolute directory or an absolute file path. */
function normalizeModuleSpecifier(candidate) {
  if (candidate.startsWith("file://")) return candidate;
  if (!candidate.includes("/") && !candidate.includes("\\")) return candidate;
  const resolved = path.resolve(candidate);
  try {
    if (fs.statSync(resolved).isDirectory()) {
      return pathToFileURL(path.join(resolved, "index.js")).href;
    }
  } catch {
    return pathToFileURL(resolved).href;
  }
  return pathToFileURL(resolved).href;
}

function markNotRun(reason, extra = {}) {
  runtime.reason = reason;
  const lines = [
    `status=NOT_RUN`,
    `mode=${MODE}`,
    `reason=${reason}`,
    `page_root=${PAGE_ROOT}`,
    `node=${process.version}`,
    `os=${process.platform}`,
    `cwd=${process.cwd()}`,
    ...Object.entries(extra).map(([key, value]) => `${key}=${value}`),
    "",
    "This file records that the UI suite did not observe the page.",
    "A NOT_RUN result must not be reported as a pass.",
  ];
  try {
    fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
    fs.writeFileSync(NOT_RUN_FILE, lines.join("\n"), "utf8");
  } catch {
    /* evidence file is best effort */
  }
  console.log(`[NOT_RUN] ${reason}`);
}

function requireRuntime(t) {
  if (runtime.browser && runtime.baseUrl) return true;
  t.skip(`NOT_RUN: ${runtime.reason || "browser or Host unavailable"}`);
  return false;
}

/* ─────────────────────────── static file server ────────────────────────── */

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

function startStaticServer(root) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((request, response) => {
      const url = new URL(request.url || "/", "http://127.0.0.1");
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === "/" || pathname.endsWith("/")) pathname += "index.html";
      const target = path.resolve(root, `.${pathname}`);
      if (!target.startsWith(root)) {
        response.writeHead(403).end("forbidden");
        return;
      }
      fs.readFile(target, (error, buffer) => {
        if (error) {
          response.writeHead(404, { "content-type": "text/plain" }).end("not found");
          return;
        }
        response.writeHead(200, {
          "content-type": MIME[path.extname(target)] || "application/octet-stream",
          "cache-control": "no-store",
        });
        response.end(buffer);
      });
    });
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({ server, url: `http://127.0.0.1:${port}` });
    });
  });
}

/* ─────────────────────────── fixture backend ───────────────────────────── */

const APPLY_SCOPE = {
  upper_enabled: "next_input",
  routing_provider_id: "next_input",
  generation_provider_id: "next_input",
  routing_prompt: "next_input",
  planning_prompt: "next_input",
  model_timeout_s: "next_input",
  control_enabled: "reload_required",
  control_bind_host: "reload_required",
  control_port: "reload_required",
  control_peer_id: "reload_required",
  robot_id: "reload_required",
};

function defaultSettings() {
  return {
    upper_enabled: false,
    routing_provider_id: "",
    generation_provider_id: "",
    routing_prompt: "",
    planning_prompt: "",
    control_enabled: false,
    control_bind_host: "127.0.0.1",
    control_port: 8769,
    control_peer_id: "",
    robot_id: "",
    model_timeout_s: 30,
  };
}

function planFixture(planId, { segments = 3, sent = 1, phase = "injected" } = {}) {
  return {
    plan_id: planId,
    route: "speech_and_action",
    input_key: `input-${planId}`,
    plan_revision: 1,
    sent_prefix: sent,
    prompt_version: "prompt-abcdef123456",
    catalog_version: "cat-abcdef123456",
    created_at: "2026-11-02T06:00:00Z",
    delivery_phase: phase,
    segments: Array.from({ length: segments }, (_value, index) => ({
      segment_id: `seg-${index}`,
      ordinal: index,
      text: `第 ${index + 1} 段：寻找红色杯子并对齐同一个目标。`,
    })),
    deliveries: [
      {
        batch_id: "batch-1",
        phase,
        payload_hash: "deadbeefdeadbeef",
        segment_ids: ["seg-0"],
        attempted_at: "2026-11-02T06:00:02Z",
      },
    ],
  };
}

function createFixture() {
  return {
    settings: defaultSettings(),
    settingsRevision: 1,
    records: [],
    catalogRevision: 1,
    catalogVersion: "cat-1",
    nextRef: 1,
    plans: [planFixture("plan-1")],
    exState: "offline",
    failures: [],
    requests: [],
  };
}

function fail(fixture, endpoint, status, code) {
  fixture.failures.push({ endpoint, status, code });
}

function takeFailure(fixture, endpoint) {
  const index = fixture.failures.findIndex((item) => item.endpoint === endpoint);
  if (index === -1) return null;
  return fixture.failures.splice(index, 1)[0];
}

function reject(code, text) {
  return { ok: false, message: `[UPPER:${code}] ${text}` };
}

function providerRows() {
  return [
    { id: "prov-a", name: "主模型", model: "gpt-x", type: "openai", available: true, reason: null },
    { id: "prov-b", name: "备用模型", model: "claude-y", type: "anthropic", available: true, reason: null },
  ];
}

function handleRequest(fixture, method, endpoint, params, body) {
  const injected = takeFailure(fixture, endpoint);
  if (injected) return reject(injected.code, "注入的失败");

  const query = params || {};
  const payload = body || {};

  if (method === "GET" && endpoint === "upper/settings") {
    return {
      ok: true,
      data: {
        settings: { ...fixture.settings },
        revision: fixture.settingsRevision,
        apply_scope: { ...APPLY_SCOPE },
        sources: {},
        runtime: {
          control_state: "stopped",
          control_enabled: fixture.settings.control_enabled,
          control_peer: fixture.settings.control_peer_id || null,
          ex_session: null,
          input_ready: false,
          last_delivery: null,
        },
        fields: Object.keys(APPLY_SCOPE).map((name) => ({
          name,
          type: "string",
          apply_scope: APPLY_SCOPE[name],
        })),
      },
    };
  }

  if (method === "POST" && endpoint === "upper/settings/save") {
    if (payload.expected_revision !== fixture.settingsRevision) {
      return reject("revision_conflict", "设置已被其他页面更新");
    }
    const changed = [];
    for (const [key, value] of Object.entries(payload.settings || {})) {
      if (!(key in APPLY_SCOPE)) return reject("field_not_allowed", `字段 ${key} 不接受`);
      if (fixture.settings[key] !== value) changed.push(key);
      fixture.settings[key] = value;
    }
    fixture.settingsRevision += 1;
    return {
      ok: true,
      data: {
        revision: fixture.settingsRevision,
        changed,
        reload_required: changed.filter((key) => APPLY_SCOPE[key] === "reload_required"),
        next_input: changed.filter((key) => APPLY_SCOPE[key] === "next_input"),
      },
    };
  }

  if (method === "GET" && endpoint === "upper/providers") {
    return {
      ok: true,
      data: {
        providers: providerRows(),
        selected: {
          routing_provider_id: fixture.settings.routing_provider_id,
          generation_provider_id: fixture.settings.generation_provider_id,
        },
        stale: [],
        installed: true,
      },
    };
  }

  if (method === "GET" && endpoint === "upper/actions") {
    return {
      ok: true,
      data: {
        actions: fixture.records.map((record) => ({ ...record })),
        revision: fixture.catalogRevision,
        catalog_version: fixture.catalogVersion,
        ex_availability: {
          state: fixture.exState,
          catalog_version: fixture.exState === "online" ? "ex-1" : null,
          action_ids: fixture.exState === "online" ? ["ex.pick_object"] : [],
          checked_at: "2026-11-02T06:00:00Z",
        },
      },
    };
  }

  if (method === "POST" && endpoint === "upper/actions/save") {
    if (payload.expected_revision !== fixture.catalogRevision) {
      return reject("revision_conflict", "动作目录已被其他页面更新");
    }
    const record = payload.record || {};
    if (!record.name) return reject("invalid_request", "名称不能为空");
    if (record.level !== "skill" && !record.action_id) {
      return reject("invalid_request", "primitive/basic 需要 action_id");
    }
    let actionRef = record.action_ref;
    if (!actionRef) {
      actionRef = `aeb-${String(fixture.nextRef).padStart(3, "0")}`;
      fixture.nextRef += 1;
    }
    fixture.catalogRevision += 1;
    fixture.catalogVersion = `cat-${fixture.catalogRevision}`;
    const saved = { ...record, action_ref: actionRef, revision: fixture.catalogRevision };
    const index = fixture.records.findIndex((item) => item.action_ref === actionRef);
    if (index === -1) fixture.records.push(saved);
    else fixture.records[index] = saved;
    return {
      ok: true,
      data: {
        action_ref: actionRef,
        revision: fixture.catalogRevision,
        catalog_version: fixture.catalogVersion,
        action: saved,
      },
    };
  }

  if (method === "POST" && endpoint === "upper/actions/delete") {
    if (payload.expected_revision !== fixture.catalogRevision) {
      return reject("revision_conflict", "动作目录已被其他页面更新");
    }
    const index = fixture.records.findIndex((item) => item.action_ref === payload.action_ref);
    if (index === -1) return reject("not_found", "未找到该动作条目");
    fixture.records.splice(index, 1);
    fixture.catalogRevision += 1;
    fixture.catalogVersion = `cat-${fixture.catalogRevision}`;
    return {
      ok: true,
      data: {
        action_ref: payload.action_ref,
        deleted: true,
        revision: fixture.catalogRevision,
        catalog_version: fixture.catalogVersion,
        ex_action_untouched: true,
      },
    };
  }

  if (method === "GET" && endpoint === "upper/prompts") {
    const prompt = (kind, field) => {
      const override = fixture.settings[field] || "";
      const builtin = kind === "routing" ? "内置路由规范" : "内置编排规范";
      const effective = override || builtin;
      return {
        kind,
        content: override,
        effective_content: effective,
        source: override ? "override" : "builtin",
        hash: `${kind}-hash-${effective.length}`,
        default_hash: `${kind}-hash-${builtin.length}`,
        bytes: effective.length,
      };
    };
    return {
      ok: true,
      data: {
        revision: fixture.settingsRevision,
        prompts: {
          routing: prompt("routing", "routing_prompt"),
          planning: prompt("planning", "planning_prompt"),
        },
      },
    };
  }

  if (method === "POST" && endpoint === "upper/prompts/save") {
    if (payload.expected_revision !== fixture.settingsRevision) {
      return reject("revision_conflict", "设置已被其他页面更新");
    }
    const field = payload.kind === "routing" ? "routing_prompt" : "planning_prompt";
    if (!(payload.kind === "routing" || payload.kind === "planning")) {
      return reject("invalid_request", "kind 只接受 routing 或 planning");
    }
    fixture.settings[field] = payload.content || "";
    fixture.settingsRevision += 1;
    return {
      ok: true,
      data: {
        kind: payload.kind,
        revision: fixture.settingsRevision,
        source: payload.content ? "override" : "builtin",
        hash: `${payload.kind}-hash`,
        bytes: (payload.content || "").length,
        reload_required: [],
      },
    };
  }

  if (method === "GET" && endpoint === "upper/plans") {
    const limit = Number(query.limit || 20);
    const offset = query.cursor ? Number(Buffer.from(query.cursor, "base64").toString("ascii")) : 0;
    const window = fixture.plans.slice(offset, offset + limit);
    const hasMore = offset + limit < fixture.plans.length;
    return {
      ok: true,
      data: {
        plans: window.map((plan) => ({
          plan_id: plan.plan_id,
          route: plan.route,
          input_key: plan.input_key,
          plan_revision: plan.plan_revision,
          segment_total: plan.segments.length,
          sent_prefix: plan.sent_prefix,
          pending_suffix: plan.segments.length - plan.sent_prefix,
          delivery_phase: plan.delivery_phase,
          prompt_version: plan.prompt_version,
          catalog_version: plan.catalog_version,
          created_at: plan.created_at,
        })),
        limit,
        next_cursor: hasMore
          ? Buffer.from(String(offset + window.length)).toString("base64")
          : null,
        count: window.length,
      },
    };
  }

  if (method === "GET" && endpoint === "upper/plans/get") {
    const plan = fixture.plans.find((item) => item.plan_id === query.plan_id);
    if (!plan) return reject("not_found", "未找到该计划");
    return {
      ok: true,
      data: {
        plan: {
          ...plan,
          segment_total: plan.segments.length,
          pending_suffix: plan.segments.length - plan.sent_prefix,
          pending_segments: plan.segments.slice(plan.sent_prefix),
          visibility: "decision_only",
        },
      },
    };
  }

  return reject("not_found", `未实现 ${method} ${endpoint}`);
}

/* ─────────────────────────── page helpers ──────────────────────────────── */

async function openFakePage(browser, fixture, options = {}) {
  const context = await browser.newContext({
    viewport: options.viewport || VIEWPORT_WIDE,
    colorScheme: options.colorScheme || "light",
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  await page.exposeFunction("__upperHandle", async (method, endpoint, params, body) => {
    fixture.requests.push({ method, endpoint, params: params || null, body: body || null });
    return handleRequest(fixture, method, endpoint, params, body);
  });

  await page.addInitScript(() => {
    window.AstrBotPluginPage = {
      ready: async () => ({
        pluginName: "astrbot_plugin_astrbotex_interaction",
        displayName: "AstrBotEX ZeroMQ Integration",
        pageName: "upper-control",
        pageTitle: "上位控制",
        locale: "zh-CN",
        i18n: {},
        isDark: false,
      }),
      getContext: () => ({ isDark: false, locale: "zh-CN" }),
      getLocale: () => "zh-CN",
      getI18n: () => ({}),
      t: (_key, fallback) => fallback,
      onContext: () => () => {},
      apiGet: async (endpoint, params) => {
        const result = await window.__upperHandle("GET", endpoint, params || {}, null);
        if (!result.ok) throw new Error(result.message);
        return result.data;
      },
      apiPost: async (endpoint, body) => {
        const result = await window.__upperHandle("POST", endpoint, null, body || {});
        if (!result.ok) throw new Error(result.message);
        return result.data;
      },
    };
  });

  await page.goto(`${runtime.baseUrl}/index.html`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#tab-actions");
  await page.waitForFunction(() => {
    const chip = document.getElementById("status-revision");
    return chip && chip.textContent.trim() !== "rev —";
  });
  return { context, page, errors };
}

async function openRealHostPage(browser) {
  const storage = process.env.UPPER_HOST_STORAGE;
  const cookie = process.env.UPPER_HOST_COOKIE;
  const context = await browser.newContext({
    viewport: VIEWPORT_WIDE,
    storageState: storage || undefined,
  });
  // The Host's first-use announcement is outside this plugin's acceptance scope.
  await context.addInitScript(() => {
    if (window === window.top) localStorage.setItem("astrbot:first_notice_seen:v1", "1");
  });
  if (cookie && !storage) {
    const url = new URL(process.env.UPPER_HOST_URL);
    await context.addInitScript((token) => {
      if (window === window.top) localStorage.setItem("token", token);
    }, cookie);
    await context.addCookies([
      { name: process.env.UPPER_HOST_COOKIE_NAME || "astrbot_dashboard_jwt",
        value: cookie, domain: url.hostname, path: "/" },
    ]);
  }
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  const url = `${process.env.UPPER_HOST_URL.replace(/\/$/, "")}/#/plugin-page/${PLUGIN_NAME}/upper-control`;
  await page.goto(url, { waitUntil: "domcontentloaded" });
  const iframe = await page.waitForSelector("iframe.plugin-page-frame", { timeout: 30000 });
  const frame = await iframe.contentFrame();
  assert.ok(frame, "Dashboard must create the real plugin iframe");
  await frame.waitForSelector("#tab-actions");
  await frame.waitForFunction(() => /rev \d+/.test(document.querySelector("#status-revision").textContent));
  return { context, page, frame, errors };
}

async function setTheme(page, theme) {
  await page.evaluate((value) => {
    document.documentElement.setAttribute("data-theme", value);
  }, theme);
}

async function horizontalOverflow(page) {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

async function newActionDraft(page) {
  await page.click("#btn-new-action");
  await page.waitForSelector("#action-form:not([hidden])");
}

/**
 * Screenshot plus the API fact it is evidence for.
 *
 * B08 asks for real operation screenshots *and* the matching API assertions, so
 * each capture records the request it depended on.  A screenshot failure is
 * swallowed: evidence must never turn into a false test failure.
 */
const evidence = [];

async function capture(page, file, { area, note, assertion, request }) {
  await page.waitForTimeout(60);
  let saved = false;
  try {
    fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
    await page.screenshot({ path: path.join(ARTIFACT_DIR, file) });
    saved = true;
  } catch {
    /* best effort */
  }
  evidence.push({ file: saved ? file : null, area, note, assertion, request: request || null });
  return saved;
}

function writeEvidence() {
  try {
    fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
    fs.writeFileSync(
      EVIDENCE_FILE,
      JSON.stringify(
        { mode: MODE, generated_at: new Date().toISOString(), page_root: PAGE_ROOT, entries: evidence },
        null,
        2,
      ),
      "utf8",
    );
  } catch {
    /* evidence file is best effort */
  }
}

async function fillAction(page, { name, level = "primitive", actionId = "", skill = "" }) {
  await page.selectOption("#field-level", level);
  await page.fill("#field-name", name);
  if (level === "skill") {
    await page.fill("#field-skill", skill || "# 独立 Skill\n按阶段描述即可。");
  } else {
    await page.fill("#field-action-id", actionId || "ex.pick_object");
  }
  await page.fill("#field-description", "按已配置方案执行，用于界面回归测试。");
}

/* ─────────────────────────── suite lifecycle ───────────────────────────── */

before(async () => {
  if (MODE === "real_host") {
    const host = process.env.UPPER_HOST_URL;
    if (!host) {
      markNotRun("real_host mode needs UPPER_HOST_URL");
      return;
    }
    runtime.playwright = await loadPlaywright();
    if (!runtime.playwright) {
      markNotRun("playwright module not resolvable (set PLAYWRIGHT_MODULE)");
      return;
    }
    try {
      runtime.browser = await runtime.playwright.chromium.launch();
    } catch (error) {
      markNotRun(`chromium launch failed: ${error.message}`);
      return;
    }
    runtime.baseUrl = host;
    runtime.extra.host = host;
    return;
  }

  if (!fs.existsSync(path.join(PAGE_ROOT, "index.html"))) {
    markNotRun(`page entry missing at ${PAGE_ROOT}/index.html`);
    return;
  }
  runtime.playwright = await loadPlaywright();
  if (!runtime.playwright) {
    markNotRun("playwright module not resolvable (set PLAYWRIGHT_MODULE)");
    return;
  }
  try {
    runtime.browser = await runtime.playwright.chromium.launch();
  } catch (error) {
    markNotRun(`chromium launch failed: ${error.message}`);
    return;
  }
  try {
    const started = await startStaticServer(PAGE_ROOT);
    runtime.server = started.server;
    runtime.baseUrl = started.url;
    runtime.extra.base_url = started.url;
  } catch (error) {
    markNotRun(`static server failed: ${error.message}`);
  }
});

after(async () => {
  writeEvidence();
  if (runtime.browser) await runtime.browser.close().catch(() => {});
  if (runtime.server) await new Promise((resolve) => runtime.server.close(resolve));
});

function fakeTest(name, callback) {
  if (MODE !== "real_host") test(name, callback);
}

/* ─────────────────────────── tests ────────────────────────────────────── */

fakeTest("boots with four areas and the runtime status chips", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  const { context, page, errors } = await openFakePage(runtime.browser, fixture);
  try {
    assert.equal(await page.locator(".tab").count(), 4);
    assert.equal(await page.textContent("#status-revision"), "rev 1");
    assert.match(await page.textContent("#status-ex-text"), /EX 离线/);
    assert.match(await page.textContent("#status-control-text"), /通道/);
    assert.equal(await page.locator("#action-empty").isVisible(), true);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

fakeTest("creates an action, saves it, and it survives a reload", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  let session = await openFakePage(runtime.browser, fixture);
  try {
    await newActionDraft(session.page);
    await fillAction(session.page, { name: "抓取红色杯子" });
    await session.page.click("#btn-add-param");
    await session.page.fill('[data-role="param-name"]', "target");
    await session.page.fill('[data-role="param-description"]', "目标引用");
    await session.page.fill('[data-role="param-unit"]', "");
    await session.page.check('[data-role="param-required"]');
    await session.page.click("#btn-save-action");
    await session.page.waitForFunction(() =>
      document.getElementById("save-state").textContent.includes("已保存"),
    );

    const saved = fixture.records.at(-1);
    assert.equal(saved.name, "抓取红色杯子");
    assert.equal(saved.action_id, "ex.pick_object");
    assert.equal(saved.parameters.length, 1);
    assert.equal(saved.parameters[0].required, true);
    assert.equal(fixture.records.length, 1);

    const saveRequest = fixture.requests.find((item) => item.endpoint === "upper/actions/save");
    assert.equal(saveRequest.body.expected_revision, 1);

    await session.context.close();
    session = await openFakePage(runtime.browser, fixture);
    await session.page.waitForSelector(".action-item");
    assert.equal(await session.page.locator(".action-item").count(), 1);
    assert.match(await session.page.textContent(".action-item-name"), /抓取红色杯子/);
  } finally {
    await session.context.close();
  }
});

fakeTest("edits, disables and deletes an entry", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  const { context, page } = await openFakePage(runtime.browser, fixture);
  try {
    await newActionDraft(page);
    await fillAction(page, { name: "接近目标" });
    await page.click("#btn-save-action");
    await page.waitForFunction(() =>
      document.getElementById("save-state").textContent.includes("已保存"),
    );

    await page.click(".action-item-button");
    await page.fill("#field-name", "接近并对齐目标");
    await page.uncheck("#field-enabled");
    await page.click("#btn-save-action");
    await page.waitForFunction(() =>
      document.getElementById("save-state").textContent.includes("已保存"),
    );
    assert.equal(fixture.records[0].name, "接近并对齐目标");
    assert.equal(fixture.records[0].enabled, false);
    assert.equal(await page.getAttribute(".action-item", "data-enabled"), "false");

    await page.click("#btn-delete-action");
    await page.waitForSelector("#confirm-dialog[open]");
    assert.match(await page.textContent("#confirm-body"), /不会卸载|EX/);
    await page.click("#confirm-ok");
    await page.waitForFunction(() => document.querySelectorAll(".action-item").length === 0);
    assert.equal(fixture.records.length, 0);
    assert.equal(await page.locator("#editor-empty").isVisible(), true);
  } finally {
    await context.close();
  }
});

fakeTest("skill level hides the executable id and previews text safely", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  fixture.exState = "online";
  const { context, page } = await openFakePage(runtime.browser, fixture);
  try {
    await newActionDraft(page);
    await page.selectOption("#field-level", "skill");
    assert.equal(await page.isDisabled("#field-action-id"), true);
    await page.fill("#field-name", "独立 Skill");
    const markdown = "# 标题\n<script>window.__xss=1</script>\n| a | b |";
    await page.fill("#field-skill", markdown);
    await page.click("#btn-preview-skill");
    await page.waitForSelector("#preview-skill:not([hidden])");
    assert.equal(await page.textContent("#preview-skill"), markdown);
    assert.equal(await page.evaluate(() => window.__xss), undefined);
    assert.equal(await page.locator("#preview-skill script").count(), 0);
  } finally {
    await context.close();
  }
});

fakeTest("prompts save independently and report their source", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  const { context, page } = await openFakePage(runtime.browser, fixture);
  try {
    await page.click("#tab-prompts");
    await page.waitForSelector("#panel-prompts:not([hidden])");
    assert.match(await page.textContent('[data-kind="routing"] [data-role="source"]'), /内置/);

    await page.fill("#prompt-routing", "只在确认目标后执行。");
    await page.click('[data-role="save"][data-kind="routing"]');
    await page.waitForFunction(() => document.getElementById("toast").textContent.includes("路由规范"));

    assert.equal(fixture.settings.routing_prompt, "只在确认目标后执行。");
    assert.equal(fixture.settings.planning_prompt, "");
    const saveRequest = fixture.requests.find((item) => item.endpoint === "upper/prompts/save");
    assert.deepEqual(saveRequest.body, {
      kind: "routing",
      content: "只在确认目标后执行。",
      expected_revision: 1,
    });

    await page.click("#tab-prompts");
    assert.match(
      await page.textContent('[data-kind="routing"] [data-role="source"]'),
      /覆盖/,
    );
  } finally {
    await context.close();
  }
});

fakeTest("provider and channel saves flag the fields that need a reload", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  const { context, page } = await openFakePage(runtime.browser, fixture);
  try {
    await page.click("#tab-settings");
    await page.waitForSelector("#panel-settings:not([hidden])");
    assert.equal(await page.locator("#field-routing-provider option").count(), 3);
    assert.equal(await page.locator('input[type="password"]').count(), 0);

    await page.selectOption("#field-routing-provider", "prov-b");
    await page.fill("#field-port", "8790");
    await page.check("#field-upper-enabled");
    await page.click("#btn-save-settings");
    await page.waitForFunction(() =>
      document.getElementById("settings-save-state").textContent.includes("已保存"),
    );

    assert.equal(fixture.settings.routing_provider_id, "prov-b");
    assert.equal(fixture.settings.control_port, 8790);
    assert.equal(fixture.settings.upper_enabled, true);
    assert.equal(await page.locator("#reload-badge").isVisible(), true);
    const note = await page.textContent("#settings-save-state");
    assert.match(note, /control_port/);
    assert.doesNotMatch(note, /upper_enabled/);
    // Saving settings performs exactly one write and never triggers a control
    // or model call from the page.
    const writes = fixture.requests.filter((item) => item.method === "POST");
    assert.deepEqual(
      writes.map((item) => item.endpoint),
      ["upper/settings/save"],
    );
  } finally {
    await context.close();
  }
});

fakeTest("revision conflict keeps the draft and offers reload or copy", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  const first = await openFakePage(runtime.browser, fixture);
  const second = await openFakePage(runtime.browser, fixture);
  try {
    await newActionDraft(first.page);
    await fillAction(first.page, { name: "先保存的条目" });
    await first.page.click("#btn-save-action");
    await first.page.waitForFunction(() =>
      document.getElementById("save-state").textContent.includes("已保存"),
    );

    await newActionDraft(second.page);
    await fillAction(second.page, { name: "后保存的条目" });
    await second.page.click("#btn-save-action");
    await second.page.waitForSelector("#banner:not([hidden])");

    const banner = await second.page.textContent("#banner-text");
    assert.match(banner, /其他页面更新/);
    assert.equal(await second.page.isVisible("#banner-primary"), true);
    assert.equal(await second.page.isVisible("#banner-secondary"), true);
    // The draft must still be in the form.
    assert.equal(await second.page.inputValue("#field-name"), "后保存的条目");
    assert.equal(await second.page.textContent("#save-state"), "保存失败，草稿已保留。");
    assert.equal(fixture.records.length, 1);
  } finally {
    await first.context.close();
    await second.context.close();
  }
});

fakeTest("503 and network failures keep every input", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  const { context, page } = await openFakePage(runtime.browser, fixture);
  try {
    await newActionDraft(page);
    await fillAction(page, { name: "依赖不可用时的草稿" });
    await page.fill("#field-description", "这段文字在失败后必须还在。");
    fail(fixture, "upper/actions/save", 503, "dependency_unavailable");
    await page.click("#btn-save-action");
    await page.waitForSelector("#banner:not([hidden])");

    assert.match(await page.textContent("#banner-text"), /依赖/);
    assert.equal(await page.inputValue("#field-name"), "依赖不可用时的草稿");
    assert.equal(await page.inputValue("#field-description"), "这段文字在失败后必须还在。");
    assert.equal(await page.isEnabled("#btn-save-action"), true);
  } finally {
    await context.close();
  }
});

fakeTest("records show delivery phases without claiming execution", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  fixture.plans = [
    planFixture("plan-sentinel", { segments: 3, sent: 1, phase: "injected" }),
    planFixture("plan-unknown", { phase: "delivery_unknown" }),
  ];
  const { context, page } = await openFakePage(runtime.browser, fixture);
  try {
    await page.click("#tab-records");
    await page.waitForSelector("#panel-records:not([hidden])");
    await page.waitForSelector(".record-row");

    assert.equal(await page.locator(".record-row").count(), 2);
    const phases = await page.locator(".record-row .cell-phase").allTextContents();
    assert.deepEqual(phases, ["已注入", "未知"]);

    await page.click(".record-row");
    await page.waitForSelector("#record-detail:not([hidden])");
    assert.equal(await page.locator("#record-segments li").count(), 3);
    assert.equal(await page.locator('#record-segments li[data-pending="true"]').count(), 2);

    // No badge may claim completion, and the detail view offers no action.
    const badges = await page.locator(".badge").allTextContents();
    assert.ok(
      badges.every((label) => !/执行完成|已完成|成功/.test(label)),
      `unexpected completion label: ${badges.join(",")}`,
    );
    assert.match(await page.textContent("#panel-records .panel-intro"), /不代表执行完成/);
    assert.equal(await page.locator("#record-detail button").count(), 0);
    assert.equal(await page.locator("#panel-records button:visible").count(), 0);

    // The sentinel text stays inside the local records area.
    const outside = await page.evaluate(() => {
      const marker = "寻找红色杯子";
      const records = document.getElementById("panel-records");
      return Array.from(document.querySelectorAll("body *"))
        .filter((node) => node.children.length === 0 && node.textContent.includes(marker))
        .filter((node) => !records.contains(node)).length;
    });
    assert.equal(outside, 0);
  } finally {
    await context.close();
  }
});

fakeTest("layout holds at 1280px and 390px, in both themes", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  fixture.records.push({
    action_ref: "aeb-900",
    action_id: "ex.pick_object",
    level: "primitive",
    name: "超长名称：" + "抓取红色杯子并保持目标引用".repeat(6),
    description: "很长的中文说明".repeat(40),
    parameters: [{ name: "target", description: "目标引用", unit: "", required: true }],
    usage_stages: ["定位", "接近", "抓取"],
    skill_markdown:
      "| 列 | 说明 |\n| --- | --- |\n" +
      Array.from({ length: 60 }, (_value, index) => `第 ${index + 1} 行：长内容用于验证滚动与换行。`).join("\n"),
    enabled: true,
    revision: 2,
  });
  const wide = await openFakePage(runtime.browser, fixture, { viewport: VIEWPORT_WIDE });
  try {
    await wide.page.waitForSelector(".action-item");
    await wide.page.click(".action-item-button");
    assert.ok((await horizontalOverflow(wide.page)) <= 1, "1280px must not scroll sideways");

    const textColorLight = await wide.page.evaluate(
      () => getComputedStyle(document.body).color,
    );
    await setTheme(wide.page, "dark");
    const darkColors = await wide.page.evaluate(() => ({
      background: getComputedStyle(document.body).backgroundColor,
      color: getComputedStyle(document.body).color,
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    }));
    assert.notEqual(darkColors.color, textColorLight);
    const rgb = darkColors.color.match(/\d+/g).map(Number);
    assert.ok(rgb.slice(0, 3).every((channel) => channel > 150), "dark theme text must be light");
    const bg = darkColors.background.match(/\d+/g).map(Number);
    assert.ok(bg.slice(0, 3).every((channel) => channel < 80), "dark theme surface must be dark");
    assert.ok(darkColors.overflow <= 1);

    const skillScrollable = await wide.page.evaluate(() => {
      const node = document.getElementById("field-skill");
      return {
        horizontal: node.scrollWidth - node.clientWidth,
        vertical: node.scrollHeight - node.clientHeight,
      };
    });
    assert.ok(skillScrollable.horizontal <= 1, "skill textarea must not scroll sideways");
    assert.ok(skillScrollable.vertical > 0, "long skill text must scroll inside the box");
  } finally {
    await wide.context.close();
  }

  const narrow = await openFakePage(runtime.browser, fixture, { viewport: VIEWPORT_NARROW });
  try {
    await narrow.page.waitForSelector(".action-item");
    await narrow.page.click(".action-item-button");
    assert.ok((await horizontalOverflow(narrow.page)) <= 1, "390px must not scroll sideways");

    const stacked = await narrow.page.evaluate(() => {
      const list = document.querySelector(".list-col").getBoundingClientRect();
      const editor = document.querySelector(".editor-col").getBoundingClientRect();
      return editor.top >= list.bottom - 1;
    });
    assert.equal(stacked, true, "narrow layout stacks list above editor");

    const tapTarget = await narrow.page.evaluate(
      () => document.getElementById("btn-save-action").getBoundingClientRect().height,
    );
    assert.ok(tapTarget >= 40, `touch target ${tapTarget}px is too small`);
  } finally {
    await narrow.context.close();
  }
});

fakeTest("unsaved changes are guarded before switching areas", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  const { context, page } = await openFakePage(runtime.browser, fixture);
  try {
    await newActionDraft(page);
    await page.fill("#field-name", "还没保存");
    await page.click("#tab-prompts");
    await page.waitForSelector("#confirm-dialog[open]");
    await page.click("#confirm-cancel");
    assert.equal(await page.isVisible("#panel-actions"), true);
    assert.equal(await page.inputValue("#field-name"), "还没保存");

    await page.click("#tab-prompts");
    await page.waitForSelector("#confirm-dialog[open]");
    await page.click("#confirm-ok");
    await page.waitForSelector("#panel-prompts:not([hidden])");
  } finally {
    await context.close();
  }
});

fakeTest("captures the operator walkthrough with its API assertions", async (t) => {
  if (!requireRuntime(t)) return;
  const fixture = createFixture();
  fixture.plans = [
    planFixture("plan-sentinel", { segments: 3, sent: 1, phase: "injected" }),
    planFixture("plan-unknown", { phase: "delivery_unknown" }),
  ];

  const wide = await openFakePage(runtime.browser, fixture, { viewport: VIEWPORT_WIDE });
  try {
    await capture(wide.page, "01-actions-empty.png", {
      area: "actions",
      note: "空动作目录，右侧显示空状态",
      assertion: "GET upper/actions 返回 actions=[] 且 revision=1",
      request: fixture.requests.at(-1),
    });

    await newActionDraft(wide.page);
    await fillAction(wide.page, { name: "抓取红色杯子" });
    await wide.page.click("#btn-add-param");
    await wide.page.fill('[data-role="param-name"]', "target");
    await wide.page.fill('[data-role="param-description"]', "目标引用");
    await wide.page.fill('[data-role="param-unit"]', "");
    await wide.page.check('[data-role="param-required"]');
    await capture(wide.page, "02-action-draft.png", {
      area: "actions",
      note: "新建条目并填写参数，尚未保存",
      assertion: "草稿仅存在于表单，未产生任何 POST",
      request: null,
    });
    assert.equal(fixture.requests.filter((item) => item.method === "POST").length, 0);

    await wide.page.click("#btn-save-action");
    await wide.page.waitForFunction(() =>
      document.getElementById("save-state").textContent.includes("已保存"),
    );
    const saveRequest = fixture.requests.find((item) => item.endpoint === "upper/actions/save");
    assert.equal(saveRequest.body.expected_revision, 1);
    assert.equal(fixture.records.length, 1);
    await capture(wide.page, "03-action-saved.png", {
      area: "actions",
      note: "保存后条目出现在列表并保持选中",
      assertion: "POST upper/actions/save 带 expected_revision=1，目录 revision 变为 2",
      request: saveRequest,
    });

    await wide.page.click("#btn-delete-action");
    await wide.page.waitForSelector("#confirm-dialog[open]");
    await capture(wide.page, "04-delete-confirm.png", {
      area: "actions",
      note: "删除前的原生确认对话框，说明不影响 EX",
      assertion: "确认框出现时不发送请求",
      request: null,
    });
    await wide.page.click("#confirm-cancel");

    await wide.page.click("#tab-prompts");
    await wide.page.waitForSelector("#panel-prompts:not([hidden])");
    await wide.page.fill("#prompt-routing", "只在确认目标后执行。");
    await wide.page.click('[data-role="save"][data-kind="routing"]');
    await wide.page.waitForFunction(() => document.getElementById("toast").textContent.includes("路由规范"));
    const promptRequest = fixture.requests.find((item) => item.endpoint === "upper/prompts/save");
    assert.deepEqual(promptRequest.body, {
      kind: "routing",
      content: "只在确认目标后执行。",
      expected_revision: 1,
    });
    assert.equal(fixture.settings.planning_prompt, "");
    await capture(wide.page, "05-prompts.png", {
      area: "prompts",
      note: "路由规范单独保存，编排规范保持内置",
      assertion: "POST upper/prompts/save 只提交 routing，planning_prompt 仍为空",
      request: promptRequest,
    });

    await wide.page.click("#tab-settings");
    await wide.page.waitForSelector("#panel-settings:not([hidden])");
    await wide.page.selectOption("#field-routing-provider", "prov-b");
    await wide.page.fill("#field-port", "8790");
    await wide.page.click("#btn-save-settings");
    await wide.page.waitForFunction(() =>
      document.getElementById("settings-save-state").textContent.includes("已保存"),
    );
    assert.equal(fixture.settings.control_port, 8790);
    const settingsRequest = fixture.requests.find((item) => item.endpoint === "upper/settings/save");
    await capture(wide.page, "06-settings.png", {
      area: "settings",
      note: "保存后重载提示只列出 reload_required 字段",
      assertion: "POST upper/settings/save 返回 reload_required 含 control_port",
      request: settingsRequest,
    });

    await wide.page.click("#tab-records");
    await wide.page.waitForSelector(".record-row");
    await wide.page.click(".record-row");
    await wide.page.waitForSelector("#record-detail:not([hidden])");
    const badges = await wide.page.locator(".badge").allTextContents();
    assert.ok(badges.every((label) => !/执行完成|已完成/.test(label)));
    await capture(wide.page, "07-records-detail.png", {
      area: "records",
      note: "投递阶段与待发段，无任何完成字样或操作按钮",
      assertion: "GET upper/plans/get 返回 phase=injected 且 pending_segments 非空",
      request: fixture.requests.filter((item) => item.endpoint === "upper/plans/get").at(-1),
    });

    await setTheme(wide.page, "dark");
    await wide.page.click("#tab-actions");
    await wide.page.waitForSelector("#panel-actions:not([hidden])");
    assert.ok((await horizontalOverflow(wide.page)) <= 1);
    await capture(wide.page, "08-dark-actions.png", {
      area: "actions",
      note: "暗色主题下动作区无横向溢出",
      assertion: "scrollWidth - clientWidth <= 1",
      request: null,
    });
  } finally {
    await wide.context.close();
  }

  const narrow = await openFakePage(runtime.browser, fixture, { viewport: VIEWPORT_NARROW });
  try {
    await setTheme(narrow.page, "dark");
    await narrow.page.click("#tab-records");
    await narrow.page.waitForSelector(".record-row");
    await narrow.page.click(".record-row");
    await narrow.page.waitForSelector("#record-detail:not([hidden])");
    assert.ok((await horizontalOverflow(narrow.page)) <= 1, "390px must not scroll sideways");
    const tapTarget = await narrow.page.evaluate(
      () => document.getElementById("tab-actions").getBoundingClientRect().height,
    );
    assert.ok(tapTarget >= 40, `touch target ${tapTarget}px is too small`);
    await capture(narrow.page, "09-narrow-dark-records.png", {
      area: "records",
      note: "390px 暗色主题，记录详情纵向堆叠且无横向溢出",
      assertion: "scrollWidth - clientWidth <= 1 且标签高度 >= 40px",
      request: null,
    });
  } finally {
    await narrow.context.close();
  }

  assert.equal(evidence.length, 9);
  assert.ok(evidence.every((entry) => entry.file), "every walkthrough step must produce a screenshot");
});

if (MODE === "real_host") test("real Host page: configure offline, persist, conflict and preserve failed drafts", async (t) => {
  if (!requireRuntime(t)) return;
  const first = await openRealHostPage(runtime.browser);
  let second;
  const endpoint = `${process.env.UPPER_HOST_URL.replace(/\/$/, "")}/api/v1/plugins/extensions/${PLUGIN_NAME}`;
  try {
    const settings = await first.frame.evaluate(async () => {
      const response = await window.AstrBotPluginPage.apiGet("upper/settings");
      return response;
    });
    assert.equal(typeof settings.revision, "number");
    assert.equal(settings.settings.upper_enabled, false, "use a disposable, feature-off Host");
    assert.equal(settings.runtime.control_state, "stopped");

    const anonymous = await runtime.browser.newContext();
    try {
      for (const path of ["upper/settings", "upper/actions", "upper/prompts", "upper/plans"]) {
        const response = await anonymous.request.get(`${endpoint}/${path}`);
        assert.ok([401, 403].includes(response.status()), `anonymous ${path} must fail`);
      }
      const response = await anonymous.request.post(`${endpoint}/upper/settings/save`, {
        data: { expected_revision: settings.revision, settings: { upper_enabled: true } },
      });
      assert.ok([401, 403].includes(response.status()), "anonymous writes must fail");
    } finally {
      await anonymous.close();
    }

    const name = `Host offline persistence ${Date.now()}`;
    await newActionDraft(first.frame);
    await fillAction(first.frame, { name, level: "skill", skill: "# Host fixture\n仅验证目录，不执行。" });
    const savedResponse = first.page.waitForResponse((r) => r.url().endsWith("/upper/actions/save") && r.request().method() === "POST");
    await first.frame.click("#btn-save-action");
    const response = await savedResponse;
    assert.equal(response.status(), 200);
    const body = await response.json();
    assert.ok(body.action_ref);
    assert.ok(!("data" in body), "Host endpoint returns the business object to its bridge");
    await first.frame.waitForFunction(() => document.querySelector("#save-state").textContent.includes("已保存"));
    await first.page.reload();
    const refreshed = await first.page.waitForSelector("iframe.plugin-page-frame");
    first.frame = await refreshed.contentFrame();
    await first.frame.waitForSelector(".action-item-name");
    assert.ok((await first.frame.locator(".action-item-name").allTextContents()).includes(name));

    await first.frame.click("#tab-settings");
    const newTimeout = settings.settings.model_timeout_s === 31 ? "32" : "31";
    await first.frame.fill("#field-timeout", newTimeout);
    await first.frame.click("#btn-save-settings");
    await first.frame.waitForFunction(() => document.querySelector("#settings-save-state").textContent.includes("已保存"));
    await first.page.reload();
    first.frame = await (await first.page.waitForSelector("iframe.plugin-page-frame")).contentFrame();
    await first.frame.click("#tab-settings");
    await first.frame.waitForFunction((value) => document.querySelector("#field-timeout").value === value, newTimeout);

    await first.frame.click("#tab-prompts");
    const prompt = `真实 Host 提示词持久化 ${Date.now()}`;
    await first.frame.fill("#prompt-routing", prompt);
    const promptSaved = first.page.waitForResponse((r) => r.url().endsWith("/upper/prompts/save"));
    await first.frame.click('[data-role="save"][data-kind="routing"]');
    assert.equal((await promptSaved).status(), 200);
    await first.page.reload();
    first.frame = await (await first.page.waitForSelector("iframe.plugin-page-frame")).contentFrame();
    await first.frame.click("#tab-prompts");
    await first.frame.waitForFunction((value) => document.querySelector("#prompt-routing").value === value, prompt);
    await first.frame.click("#tab-actions");

    // Both pages hold the same revision. Only the first save may advance it.
    second = await openRealHostPage(runtime.browser);
    await newActionDraft(first.frame);
    await fillAction(first.frame, { name: `${name} first`, level: "skill" });
    await newActionDraft(second.frame);
    await fillAction(second.frame, { name: `${name} stale`, level: "skill" });
    await first.frame.click("#btn-save-action");
    await first.frame.waitForFunction(() => document.querySelector("#save-state").textContent.includes("已保存"));
    await second.frame.click("#btn-save-action");
    await second.frame.waitForSelector("#banner:not([hidden])");
    assert.match(await second.frame.textContent("#banner-text"), /其他页面更新/);
    assert.equal(await second.frame.inputValue("#field-name"), `${name} stale`);

    // Inject only the response failure, while keeping the real iframe/bridge.
    await first.page.route("**/upper/actions/save", (route) => route.fulfill({
      status: 503, contentType: "application/json",
      body: JSON.stringify({ status: "error", message: "[UPPER:dependency_unavailable] fixture failure" }),
    }));
    await newActionDraft(first.frame);
    await fillAction(first.frame, { name: `${name} preserved`, level: "skill" });
    await first.frame.click("#btn-save-action");
    await first.frame.waitForSelector("#banner:not([hidden])");
    assert.match(await first.frame.textContent("#banner-text"), /依赖/);
    assert.equal(await first.frame.inputValue("#field-name"), `${name} preserved`);
    await first.page.unroute("**/upper/actions/save");
    await first.page.route("**/upper/actions/save", (route) => route.abort("failed"));
    await first.frame.click("#btn-save-action");
    await first.frame.waitForFunction(() => document.querySelector("#save-state").textContent.includes("草稿已保留"));
    assert.equal(await first.frame.inputValue("#field-name"), `${name} preserved`);
    assert.deepEqual(first.errors, []);
    fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
    await first.page.screenshot({ path: path.join(ARTIFACT_DIR, "real-host.png") });
    evidence.push({ file: "real-host.png", area: "real_host", note: "feature-off persistence, conflicts, anonymous refusal and failed draft retention", assertion: "Page→bridge→API→store" });
  } finally {
    if (second) await second.context.close();
    await first.context.close();
  }
});
