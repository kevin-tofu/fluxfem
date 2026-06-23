import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(__file__))
from nitsche_contact_supermesh_api import (  # noqa: E402
    NitscheContactParams,
    run_skfem_demo,
)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _params_from_env() -> NitscheContactParams:
    top_nx_pts = _env_int("SK_TOP_NX_PTS", 20)
    top_ny_pts = _env_int("SK_TOP_NY_PTS", 20)
    top_nz_pts = _env_int("SK_TOP_NZ_PTS", 10)
    bot_nx_pts = _env_int("SK_BOT_NX_PTS", 20)
    bot_ny_pts = _env_int("SK_BOT_NY_PTS", 20)
    bot_nz_pts = _env_int("SK_BOT_NZ_PTS", 5)
    quad_order = _env_int("QUAD_ORDER", 5)
    return NitscheContactParams(
        nx_top=top_nx_pts - 1,
        ny_top=top_ny_pts - 1,
        nz_top=top_nz_pts - 1,
        nx_bot=bot_nx_pts - 1,
        ny_bot=bot_ny_pts - 1,
        nz_bot=bot_nz_pts - 1,
        quad_order=quad_order,
    )


if __name__ == "__main__":
    params = _params_from_env()
    out_dir = Path(__file__).resolve().parent / "results" / "nitsche_contact_supermesh_demo_skfem"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = os.getenv("NITSCHE_SKFEM_LOG")
    npz_path = os.getenv("NITSCHE_SKFEM_U_NPZ")
    result = run_skfem_demo(
        params,
        log_path=log_path,
        npz_path=npz_path,
        vtu_path=str(out_dir / "combined-nitsche.vtu"),
        plot_path=str(out_dir / "mortar.png"),
        verbose=True,
    )
    if log_path:
        print(f"wrote summary log: {log_path}")
    if npz_path:
        print(f"wrote displacement npz: {npz_path}")
    print(f"Exported combined VTU: {out_dir / 'combined-nitsche.vtu'}")
    np.savez(out_dir / "init_from_nitsche.npz", u=result.u)
