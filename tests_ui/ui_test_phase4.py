"""Phase 4 browser test: attribution, circuit discovery + overlay, training,
report export — on the demo models."""
from playwright.sync_api import sync_playwright

SHOTS = "/tmp/claude-1000/-home-sm-synapse/1f43addd-a5ae-4317-a79c-69debd61ac46/scratchpad"
errors = []

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path="/home/sm/.cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell")
    page = browser.new_page(viewport={"width": 1720, "height": 1000})
    page.on("console", lambda m: errors.append(m.text)
            if m.type == "error" and "Failed to load resource" not in m.text else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8321")
    page.wait_for_selector("#demo-select option", state="attached")

    def load_demo(name):
        page.select_option("#demo-select", name)
        page.click("#btn-load-demo")
        page.wait_for_function(
            f"document.getElementById('session-info').textContent.includes('demo:{name}')")
        page.click('[data-tab="forward"]')

    # ---- J: attribution (text) ----
    load_demo("tiny_transformer")
    page.fill("#fw-text", "hello world")
    page.click("#btn-forward")
    page.wait_for_selector("#fw-output-card .tokbar", timeout=60000)
    page.click("#btn-attr")
    page.wait_for_selector(".attr-tok", timeout=60000)
    n = page.locator(".attr-tok").count()
    assert n == 11, f"11 token chips expected, got {n}"
    page.select_option("#attr-method", "ig")
    page.click("#btn-attr-run")
    page.wait_for_function(
        "document.getElementById('attr-view').textContent.includes('completeness')",
        timeout=120000)
    print("1. attribution: saliency chips (11 tokens) + IG with completeness line")

    # ---- J: attribution (vision) ----
    load_demo("cnn")
    page.click("#btn-forward")
    page.wait_for_selector("#fw-output-card .tokbar", timeout=60000)
    page.click("#btn-attr")
    page.wait_for_selector("#attr-view canvas.heatmap", timeout=60000)
    print("2. attribution: vision saliency heatmap rendered")

    # ---- H: circuit discovery + overlay ----
    load_demo("tiny_transformer")
    page.click("#btn-forward")   # needed for graph/trace state
    page.wait_for_selector("#fw-output-card .tokbar", timeout=60000)
    page.click('[data-tab="patch"]')
    page.fill("#patch-clean", "hello world")
    page.fill("#patch-corr", "jjjjj world")
    page.click("#btn-circuit")
    page.wait_for_selector("#circuit-result:not(.hidden)", timeout=300000)
    prog = page.locator("#circuit-progress").text_content()
    assert "done" in prog, prog
    members = page.locator("#circuit-members").text_content()
    print(f"3. circuit sweep: {prog} · {members[:70]}…")
    # lower the threshold so something is definitely selected, then overlay
    page.fill("#circuit-thresh", "10")
    page.dispatch_event("#circuit-thresh", "input")
    page.click("#btn-circuit-overlay")
    page.wait_for_selector(".gnode.in-circuit", timeout=30000)
    lit = page.locator(".gnode.in-circuit").count()
    assert lit >= 1
    print(f"4. circuit overlay: {lit} teal nodes on the graph, persists over modes")
    page.screenshot(path=f"{SHOTS}/p4_circuit.png")
    page.click('[data-tab="patch"]')
    page.click("#btn-circuit-clear")

    # ---- I: training on the toy vision task ----
    load_demo("mlp")
    page.click("#btn-forward")
    page.wait_for_selector("#fw-output-card .tokbar", timeout=60000)
    page.click('[data-tab="update"]')
    page.select_option("#opt-name", "adam")
    page.fill("#opt-lr", "0.005")
    page.fill("#train-steps", "40")
    page.fill("#train-ck", "10")
    page.click("#btn-train")
    page.wait_for_function(
        "document.getElementById('train-status').textContent.includes('done:')",
        timeout=300000)
    status = page.locator("#train-status").text_content()
    assert "checkpoints at" in status
    print(f"5. training: {status.strip()[:90]}")
    assert page.locator("#train-curve canvas").count() == 1
    # scrub to step 0, diff, restore
    page.click('[data-tab="update"]')
    page.fill("#train-scrub", "0")
    page.dispatch_event("#train-scrub", "input")
    page.click("#btn-train-diff")
    page.wait_for_selector("#train-diff-view tr[data-name]", timeout=60000)
    page.locator("#train-diff-view tr[data-name]").first.click()
    page.wait_for_selector("#train-diff-detail canvas.heatmap", timeout=30000)
    print("6. checkpoint scrub: diff vs step-0 with heatmap trio")
    page.click("#btn-train-restore")
    page.wait_for_selector("#banner", timeout=60000)
    print("7. checkpoint restore: weights rolled back, forward re-ran")
    page.screenshot(path=f"{SHOTS}/p4_training.png")

    # ---- K: report export ----
    report = page.evaluate("""async () => {
      const r = await fetch(`/api/session/${S.session.session_id}/report.md`);
      return await r.text();
    }""")
    assert "# Model X-Ray report" in report
    assert "## 1. Architecture" in report
    assert "## 8. Training run" in report, "training section missing from report"
    print(f"8. report export: {len(report)//1024}KB markdown with training section")

    browser.close()

if errors:
    print("\nJS ERRORS:")
    for e in errors:
        print("  ", e)
    raise SystemExit(1)
print("\nPHASE 4 UI TEST PASSED — no JS errors")
