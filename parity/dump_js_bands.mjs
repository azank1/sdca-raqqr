// Parity harness: the band math copied VERBATIM from the artifact, used to emit
// golden band prices for a set of dates. The Python port is asserted against this
// output in tests/test_raqqr_parity.py. If the frontend coefficients change,
// regenerate (node parity/dump_js_bands.mjs > parity/golden_bands.json) and the
// test will flag any Python drift.
import { writeFileSync } from "node:fs";

const ONE_DAY = 86400000;
const RAQQR_MU = 7.9914;
const RAQQR_GENESIS = Date.UTC(2009, 0, 1);
const RAQQR_KEYS = ["0.01", "0.1", "0.25", "0.5", "0.75", "0.95", "0.99"];
const RAQQR_COEF = {
  "0.01": [2.837, 2.578, -0.0241],
  "0.1":  [2.933, 2.552, -0.0241],
  "0.25": [3.004, 2.554, -0.0241],
  "0.5":  [3.214, 2.482, -0.1126],
  "0.75": [3.562, 2.283, -0.3259],
  "0.95": [3.897, 1.964, -0.3259],
  "0.99": [4.028, 1.904, -0.3259],
};
function raqqrPricesAtMs(ms) {
  const t = Math.max(1, Math.round((ms - RAQQR_GENESIS) / ONE_DAY));
  const x = Math.log(t) - RAQQR_MU;
  const raw = RAQQR_KEYS.map((k) => {
    const c = RAQQR_COEF[k];
    return Math.pow(10, c[0] + c[1] * x + c[2] * x * x);
  });
  return raw.slice().sort((p, q) => p - q);
}

const dates = [
  "2011-06-01", "2013-01-01", "2014-12-01", "2016-07-09", "2017-12-17",
  "2018-12-15", "2020-03-12", "2021-11-10", "2022-11-21", "2024-04-20",
  "2025-01-01", "2025-06-24", "2026-01-01",
];
const out = {};
for (const d of dates) {
  const ms = Date.parse(d + "T00:00:00Z");
  out[d] = raqqrPricesAtMs(ms);
}
writeFileSync(new URL("./golden_bands.json", import.meta.url),
  JSON.stringify(out, null, 2));
console.log("wrote golden_bands.json for", dates.length, "dates");
