"""Calibrate the Vas. Olgas car demand to Region of Attica loop detectors, then re-run the case. From the project root:
    python -m urbanintel.calibrate              # uses weekday peaks from the logger when available, else the Nov 2025 Saturday
Fixed-point iteration: demand_scale <- demand_scale x median(observed / model), written to urbanintel/config/cases.yaml."""
import re

from .services import mobility, traffic


def set_scale(x):
    p = mobility.CFG_PATH
    s = p.read_text()
    s = re.sub(r"(    demand_scale: )[0-9.]+", lambda m: f"{m.group(1)}{x:.3f}", s, count=1)
    p.write_text(s)


def main():
    n = mobility.cfg().get("calibration", {}).get("iterations", 4)
    for i in range(n):
        mobility.network.cache_clear()
        r = traffic.run_calibration()
        key = "after" if "after" in r else "before"
        m = r[key]["scale_median"]
        sc = mobility.cfg()["cars"].get("demand_scale", 1.0)
        print(f"iteration {i + 1}: scale {sc:.3f}, observed/model {m} ({key}), MAPE {r[key]['mape_pct']}%")
        if m is None or abs(m - 1) < 0.05:
            break
        set_scale(sc * m)
    print("re-running the case with the calibrated demand ...")
    mobility.run_case(force=True)
    traffic.run_calibration()


if __name__ == "__main__":
    main()
