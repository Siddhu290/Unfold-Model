"""Phase 5 browser test: steering, SAE, batch aggregation, robustness."""
from playwright.sync_api import sync_playwright

CHROME = ("/home/sm/.cache/ms-playwright/chromium_headless_shell-1223/"
          "chrome-headless-shell-linux64/chrome-headless-shell")
SHOTS = "/tmp/claude-1000/-home-sm-synapse/1f43addd-a5ae-4317-a79c-69debd61ac46/scratchpad"
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

    # ---- L: steering ----
    load_demo("tiny_transformer")
    page.fill("#fw-text", "hello world")
    page.click("#btn-forward")
    page.wait_for_selector("#fw-output-card .tokbar", timeout=60000)
    page.locator(".repeat-badge").first.click()
    page.locator(".trow", has_text="TransformerBlock").nth(2).click()  # blocks.1-ish
    page.click('[data-tab="steer"]')
    page.fill("#steer-a", "happy happy joy")
    page.fill("#steer-b", "angry angry sad")
    page.click("#btn-steer-dir")
    page.wait_for_selector("#steer-run:not(.hidden)", timeout=60000)
    info = page.locator("#steer-dir-info").text_content()
    assert "‖direction‖" in info
    page.wait_for_selector("#steer-result .card", timeout=60000)
    # move the slider -> debounced live re-steer
    page.fill("#steer-alpha", "6")
    page.dispatch_event("#steer-alpha", "input")
    page.wait_for_function(
        "document.querySelector('#steer-result .card h3').textContent.includes('α = 6')",
        timeout=60000)
    print(f"1. steering: direction built ({info.strip()[:50]}…), live α slider re-steers")
    page.fill("#steer-batch-prompts", "one two three\nfour five six")
    page.click("#btn-steer-batch")
    page.wait_for_selector("#steer-batch-result table", timeout=60000)
    rows = page.locator("#steer-batch-result tr").count()
    assert rows == 3
    print("2. steering: generalization table over 2 unrelated prompts")
    page.screenshot(path=f"{SHOTS}/p5_steer.png")

    # ---- M: SAE ----
    page.locator(".trow", has_text="TransformerBlock").nth(3).click()
    page.click('[data-tab="sae"]')
    page.fill("#sae-steps", "150")
    page.click("#btn-sae-train")
    page.wait_for_function(
        "document.getElementById('sae-status').textContent.includes('done in')",
        timeout=300000)
    status = page.locator("#sae-status").text_content()
    assert "features alive" in status
    page.click("#btn-sae-decompose")
    page.wait_for_selector("#sae-features .card", timeout=60000)
    nfeat = page.locator("#sae-features .gradbar-row").count()
    assert nfeat >= 1
    print(f"3. SAE: trained ({status.strip()[:60]}…), decomposition shows {nfeat} active features")
    page.screenshot(path=f"{SHOTS}/p5_sae.png")

    # ---- N: batch attribution ----
    page.click('[data-tab="forward"]')
    page.click("#btn-attr")
    page.wait_for_selector(".attr-tok", timeout=60000)
    page.click("#btn-attr-batch")
    page.wait_for_selector("#attr-view table", timeout=300000)
    txt = page.locator("#attr-view .card").text_content()
    assert "concentration across" in txt
    print("4. batch attribution: distribution table across prompt batch")

    # ---- N: aggregated head importance ----
    page.locator(".trow", has_text="CausalSelfAttention").first.click()
    page.wait_for_selector("#btn-ablate:not(.hidden)")
    page.click("#btn-ablate")
    page.wait_for_selector("#analysis-view .gradbar-row", timeout=120000)
    page.locator("#analysis-view button", has_text="Σ across N prompts").click()
    page.wait_for_function(
        "document.getElementById('analysis-view').textContent.includes('mean importance over')",
        timeout=300000)
    print("5. aggregated head importance: multi-prompt evidence view rendered")

    # ---- O: fragility (text) ----
    page.click("#btn-fragility")
    page.wait_for_selector("#attr-view table", timeout=300000)
    frag = page.locator("#attr-view").text_content()
    assert "fragility of" in frag
    print("6. fragility scan: per-position substitution table rendered")

    # ---- O: FGSM (vision) ----
    load_demo("cnn")
    page.click("#btn-forward")
    page.wait_for_selector("#fw-output-card .tokbar", timeout=60000)
    page.click("#btn-attr")
    page.wait_for_selector("#attr-view canvas.heatmap", timeout=60000)
    page.click("#btn-fragility")
    page.wait_for_selector("#attr-view canvas.hist", timeout=120000)
    assert "FGSM" in page.locator("#attr-view").text_content()
    print("7. FGSM sweep: degradation curve rendered for vision model")

    # report includes phase-5 sections
    report = page.evaluate("""async () => {
      const r = await fetch(`/api/session/${S.session.session_id}/report.md`);
      return await r.text();
    }""")
    assert "FGSM" in report, "robustness finding must reach the report"
    print("8. report: robustness finding serialized")

    browser.close()

if errors:
    print("\nJS ERRORS:")
    for e in errors:
        print("  ", e)
    raise SystemExit(1)
print("\nPHASE 5 UI TEST PASSED — no JS errors")
