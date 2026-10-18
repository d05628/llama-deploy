"""Exercise the manager's embedded JavaScript without starting a GPU server."""

import json
import re
import shutil
import subprocess
import unittest

import manager


@unittest.skipUnless(shutil.which("node"), "Node.js is needed for embedded UI checks")
class MultiGpuUiTests(unittest.TestCase):
    def run_ui(self, assertions):
        script = re.search(r"<script>(.*?)</script>", manager.HTML_PAGE, re.S).group(1)
        # Keep the complete script for parsing but suppress only network/timer startup.
        script = script.replace("init();startStatusPolling();", "")
        harness = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const elements = new Map();
for (const id of data.ids) {
  const classes = new Set();
  let value = '';
  elements.set(id, {
    get value() { return value; }, set value(v) { value = String(v); },
    innerHTML: '', textContent: '', style: {},
    classList: {add: c => classes.add(c), remove: c => classes.delete(c),
                contains: c => classes.has(c), toggle: c => classes.has(c) ? classes.delete(c) : classes.add(c)}
  });
}
const context = vm.createContext({
  document: {getElementById: id => elements.get(id), addEventListener() {}, querySelectorAll() {return [];}},
  window: {addEventListener() {}}, location: {hostname: 'localhost', hash: ''},
  setInterval() {}, clearInterval() {}, setTimeout() {}, clearTimeout() {}, console
});
new vm.Script(data.script).runInContext(context);
context._defaultCfg = data.defaults;
context.showToast = message => {context.lastToast = message;};
const cfg = JSON.parse(JSON.stringify(data.defaults));
const gpus = [
  {index: 0, uuid: 'GPU-zero', name: 'Card <zero>', vram_mb: 16384, vram_free_mb: 12288, free_known: true},
  {index: 1, uuid: 'GPU-one', name: 'Card one', vram_mb: 24576, vram_free_mb: 18000, free_known: true}
];
(async () => {
  await new Function('context', 'elements', 'assert', 'cfg', 'gpus',
    'return (async () => {' + data.assertions + '})();')(context, elements, assert, cfg, gpus);
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
        payload = json.dumps({
            "script": script,
            "ids": re.findall(r'\bid="([^"]+)"', manager.HTML_PAGE),
            "defaults": manager.default_config(),
            "assertions": assertions,
        })
        result = subprocess.run(
            [shutil.which("node"), "-e", harness], input=payload,
            text=True, encoding="utf-8", capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_inventory_and_primary_selection_preserve_uuid_and_escape_names(self):
        self.run_ui(r"""
context.sysInfo = {gpus};
const inventory = context.gpuInventoryHtml({gpus});
assert(inventory.includes('GPU 0') && inventory.includes('GPU 1'));
assert(inventory.includes('Card &lt;zero&gt;') && !inventory.includes('Card <zero>'));
assert(inventory.includes('12.0 GiB'));
assert(context.gpuInventoryHtml({gpus: [{...gpus[0], free_known: false}]}).includes('未知'));
context.populateMainGpuOptions('GPU-one');
assert.equal(elements.get('cfg-main_gpu').value, 'GPU-one');
assert(elements.get('cfg-main_gpu').innerHTML.includes('value="1"'));
context.populateMainGpuOptions(0);
assert.equal(elements.get('cfg-main_gpu').value, '0');
const plan = context.gpuPlanHtml({gpu_plan: {
  use_multi_gpu: true, primary_index: 1, main_gpu: 0, device_names: ['CUDA1', 'CUDA0'],
  tensor_split: [0.6, 0.4], split_mode: 'layer', notes: ['<notice>']
}});
assert(plan.includes('主 GPU: 1') && !plan.includes('主 GPU: 0'));
assert(plan.includes('0.6, 0.4') && plan.includes('&lt;notice&gt;'));
assert(context.gpuPlanHtml({gpu_plan: {main_gpu: null}}).includes('未分配 CUDA GPU'));
assert(context.gpuPlanHtml({gpu_plan: {error: '<bad config>'}}).includes('&lt;bad config&gt;'));
""")

    def test_load_save_roundtrip_and_validation_without_touching_config_file(self):
        self.run_ui(r"""
context.sysInfo = {gpus};
Object.assign(cfg.gpu, {multi_gpu: 'force', main_gpu: 'GPU-one', split_mode: 'row', tensor_split: '0,2', reserve_mb: 2048});
let saved = null, refreshed = null;
context.api = async (url, method, body) => {
  assert.equal(url, '/api/config');
  if (method === 'POST') {saved = body; return {status: 'ok'};}
  return cfg;
};
context.refreshDeviceRecommendations = async refresh => {refreshed = refresh;};
await context.loadConfig();
assert.equal(elements.get('cfg-main_gpu').value, 'GPU-one');
assert.equal(elements.get('cfg-gpu_reserve_mb').value, '2048');
await context.saveConfig();
assert(saved && refreshed === true);
for (const key of ['multi_gpu', 'main_gpu', 'split_mode', 'tensor_split', 'reserve_mb']) {
  assert.equal(saved.gpu[key], cfg.gpu[key], key);
}
saved = null; refreshed = null;
elements.get('cfg-tensor_split').value = '0,0';
await context.saveConfig();
assert.equal(saved, null); assert.equal(refreshed, null);
for (const invalid of ['-1,2', 'NaN,1', 'Infinity,1', '1,']) {
  elements.get('cfg-tensor_split').value = invalid;
  await context.saveConfig(); assert.equal(saved, null);
}
elements.get('cfg-tensor_split').value = '';
elements.get('cfg-gpu_reserve_mb').value = '-1';
await context.saveConfig(); assert.equal(saved, null);
""")

    def test_refresh_uses_saved_model_plan_and_renders_all_devices(self):
        self.run_ui(r"""
const calls = [];
context.api = async url => {
  calls.push(url);
  if (url === '/api/system') return {gpus, gpu_name: gpus[1].name, ram_gb: 64, cpu_count: 16};
  assert.equal(url, '/api/recommend');
  return {tips: ['<tip>'], gpu_max_model_gb: 16, recommended_ctx: 8192,
    gpu_plan: {use_multi_gpu: false, main_gpu: 0, primary_index: 1, device_names: ['CUDA1'], notes: []}};
};
await context.refreshDeviceRecommendations(true);
assert.deepEqual(calls, ['/api/system', '/api/recommend']);
assert(elements.get('gpuDetectInfo').innerHTML.includes('GPU 0'));
assert(elements.get('gpuDetectInfo').innerHTML.includes('GPU 1'));
assert(elements.get('gpuPlanInfo').innerHTML.includes('主 GPU: 1'));
assert(elements.get('systemInfo').innerHTML.includes('Card one'));
assert(elements.get('ramHint').innerHTML.includes('&lt;tip&gt;'));
assert.equal(context.deviceRec.recommended_ctx, 8192);
""")

    def test_market_auto_considers_multi_capacity_and_off_respects_single_budget(self):
        self.run_ui(r"""
context.sysInfo = {gpus, gpu_vram_mb: 16384, ram_gb: 64};
context.deviceRec = {gpu_max_model_gb: 10, single_gpu_max_model_gb: 10, multi_gpu_max_model_gb: 26};
context.currentConfig = {gpu: {multi_gpu: 'auto'}};
const files = [{filename: 'large.gguf', size_gb: 20, quant: 'Q4_K_M'}];
context.renderFiles(files);
assert(elements.get('fileModalContent').innerHTML.includes('预计需要多 GPU'));
context.currentConfig.gpu.multi_gpu = 'off';
context.renderFiles(files);
assert(!elements.get('fileModalContent').innerHTML.includes('预计需要多 GPU'));
assert(elements.get('fileModalContent').innerHTML.includes('超出当前 GPU 权重预算'));
context.currentConfig.gpu.multi_gpu = 'force';
for (const custom of ['1,9', [1, 9]]) {
  context.currentConfig.gpu.tensor_split = custom;
  context.renderFiles(files);
  assert(!elements.get('fileModalContent').innerHTML.includes('tag tag-green'));
  assert(elements.get('fileModalContent').innerHTML.includes('超出当前 GPU 权重预算'));
}
for (const automatic of ['', []]) {
  context.currentConfig.gpu.tensor_split = automatic;
  context.renderFiles(files);
  assert(elements.get('fileModalContent').innerHTML.includes('预计需要多 GPU'));
}
""")


if __name__ == "__main__":
    unittest.main()
