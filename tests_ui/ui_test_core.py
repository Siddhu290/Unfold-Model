"""Core browser regression: load → forward playback → inspector → backward →
optimizer step → diff → undo, plus graph view, result strip, lens, and the
patch tab, on the demo models. Run: python3.12 tests_ui/ui_test_core.py
(assumes the server is on 127.0.0.1:8321 and Playwright chromium at the
executable path below)."""
from playwright.sync_api import sync_playwright

CHROME = ("/home/sm/.cache/ms-playwright/chromium_headless_shell-1223/"
          "chrome-headless-shell-linux64/chrome-headless-shell")
errors = []

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME)
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

    # ---- forward + playback + inspector on the transformer ----
    load_demo("tiny_transformer")
    page.fill("#fw-text", "hello world")
    page.click("#btn-forward")
    page.wait_for_selector("#fw-output-card .tokbar", timeout=60000)
    assert "next token" in page.locator("#rs-verdict").text_content()
    page.click("#pb-next"); page.click("#pb-next")
    page.wait_for_selector(".trow.exec-active")
    page.locator(".trow", has_text="q_proj").first.click()
    page.wait_for_selector("#weight-view canvas.heatmap", timeout=30000)
    page.wait_for_selector(".lens-chip", timeout=30000)   # inspector logit lens
    print("1. forward, playback, inspector (weights + lens chips) OK")

    # graph view: Q/K/V fan + group sync
    page.click('[data-tab="graph"]')
    page.wait_for_selector(".gnode.group")
    page.locator(".gnode.group").first.click()
    page.wait_for_timeout(400)
    edges = page.evaluate("[...G.edgeEls.keys()]")
    assert "blocks.0.attn.q_proj→blocks.0.attn.attn_probs" in edges
    print("2. graph: group expand sync, Q/K→attn_probs edges OK")

    # lens strip
    page.click("#btn-lens")
    page.wait_for_selector(".lens-cell", timeout=60000)
    assert page.locator(".lens-cell").count() == 6
    print("3. logit lens strip: 6 cells OK")

    # backward + step + undo
    page.click('[data-tab="backward"]')
    page.click("#btn-backward")
    page.wait_for_selector("#bw-result:not(.hidden)", timeout=60000)
    assert page.locator(".gradbar-row").count() > 10
    page.click('[data-tab="update"]')
    page.click("#btn-step")
    page.wait_for_selector("#step-table tr[data-name]", timeout=60000)
    page.locator("#step-table tr[data-name]").first.click()
    page.wait_for_selector("#diff-view canvas.heatmap", timeout=30000)
    page.click("#btn-undo")
    page.wait_for_selector("#banner:not(.hidden)")
    print("4. backward bars, optimizer step diff, undo OK")

    # patch tab sanity mode on demo transformer
    page.click('[data-tab="patch"]')
    page.fill("#patch-clean", "hello world")
    page.fill("#patch-corr", "jjjjj world")
    page.select_option("#patch-pos", "all")
    page.click("#btn-patch")
    page.wait_for_selector("#patch-result:not(.hidden)", timeout=120000)
    vals = page.locator("#patch-bars .gval").all_text_contents()
    assert all(v.startswith("100.0%") for v in vals), vals
    print("5. patch sanity: 100.0% restoration at every block OK")

    # CNN skip-connection graph
    load_demo("cnn")
    page.click('[data-tab="graph"]')
    page.wait_for_selector(".gnode")
    edges = page.evaluate("[...G.edgeEls.keys()]")
    assert "relu→pool" in edges and "relu→conv2" in edges
    print("6. CNN graph: skip rejoin present OK")

    browser.close()

if errors:
    print("JS ERRORS:", *errors, sep="\n  ")
    raise SystemExit(1)
print("\nCORE UI REGRESSION PASSED — no JS errors")
