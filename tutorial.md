# fluxfem チュートリアル（scikit-fem 互換テスト付き）

このチュートリアルでは、`fluxfem` を使って

- メッシュ生成（StructuredHexBox）
- 形状関数と空間 (`make_hex_space`)
- 剛性行列/荷重ベクトルの組み立て
- SciPy の `spsolve` で解く
- scikit-fem と結果を突き合わせる

までを一気に試します。

## 0. 準備

依存関係（開発用）:

```bash
poetry add --dev pytest scikit-fem scipy
```

ローカル実行時は

```bash
PYTHONPATH=src pytest -s src/tests
```

で行列/解の比較が流れます。

## 1. メッシュと空間の構築

```python
from fluxfem import StructuredHexBox, make_hex_space

mesh = StructuredHexBox(nx=2, ny=2, nz=2, lx=1.0, ly=1.0, lz=1.0).build()
space = make_hex_space(mesh, dim=3, intorder=2)  # 3 dof/節点, Gauss-Legendre 2×2×2
```

### 境界タグ付け

軸方向の最小/最大面をタグ付けできます:

```python
from fluxfem import tag_axis_minmax_facets

facets, tags = tag_axis_minmax_facets(mesh, axis=0, dirichlet_tag=1, neumann_tag=2)
```

`intorder` は「積分の次数（polynomial exactness）」として指定します（Gauss-Legendre のテンソル積）。例:

```python
space_poor = make_hex_space(mesh, dim=1, intorder=1)  # 1×1×1 (1点) なので粗い
space_mid = make_hex_space(mesh, dim=1, intorder=3)   # 2×2×2 (8点)
space_rich = make_hex_space(mesh, dim=1, intorder=5)  # 3×3×3 (27点) で精度アップ
```
通常は 2（2×2×2, 8点）が一次ヘキサ要素のデフォルトです。

## 2. 剛性行列と荷重ベクトル

拡散・弾性などを組み立てるには `space.assemble_bilinear_form` と `space.assemble_linear_form` を使います（デバッグ用に密行列を返す内部向け `space.assemble_bilinear_dense` もあります）。

### 弾性（3D, Voigt 表記 D を渡す）

```python
import jax.numpy as jnp
from fluxfem import (
    linear_elasticity_form,
    constant_body_force_vector_form,
    isotropic_3d_D,
)

E, nu = 10.0, 0.25
D = isotropic_3d_D(E, nu)
f_vec = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)  # 体積力

K = space.assemble_bilinear_form(linear_elasticity_form, params=D)
F = space.assemble_linear_form(constant_body_force_vector_form, params=f_vec)
```

### 役割を明示した `*Spaces` family

新しいコードでは、test/trial/unknown の役割を明示したいときに
`NamedSpace` と `*Spaces` family を使えます。

```python
import fluxfem as ff
import fluxfem.helpers_wf as h_wf

U = ff.NamedSpace("U", space)
V = ff.NamedSpace("V", space)

bilinear = ff.BilinearForm.volume(
    lambda u, v, p: p.kappa * h_wf.dot(h_wf.grad(v), h_wf.grad(u)) * h_wf.dOmega()
)
linear = ff.LinearForm.volume(
    lambda v, p: (v * p.f) * h_wf.dOmega()
)
residual = ff.ResidualForm.volume(
    lambda v, u, _p: (v * (u.val**2)) * h_wf.dOmega()
)

K = ff.assemble_bilinear_form(
    ff.BilinearSpaces(test=V, trial=U),
    bilinear.get_compiled(),
    ff.Params(kappa=1.0),
)
F = ff.assemble_linear_form(
    ff.LinearSpaces(test=V),
    linear.get_compiled(),
    ff.Params(f=2.0),
)
R = ff.assemble_residual(
    ff.ResidualSpaces(test=V, unknown=U),
    residual.get_compiled(),
    jnp.zeros(space.n_dofs),
    params=None,
)
J = ff.assemble_jacobian(
    ff.JacobianSpaces(test=V, trial=U),
    residual.get_compiled(),
    jnp.zeros(space.n_dofs),
    params=None,
)
```

`U == V` の通常 Galerkin でも使えますし、将来的な `U != V` の
Petrov-Galerkin 的な書き方にもそのまま繋がります。

## 3. Dirichlet 条件を適用して解く（SciPy）

簡易的に行列修正で Dirichlet を適用し、`spsolve` で解きます。

```python
import numpy as np
from fluxfem import spsolve_numpy

coords = np.asarray(mesh.coords)
xmin = coords[:, 0].min()
dir_nodes = np.nonzero(np.isclose(coords[:, 0], xmin, atol=1e-8))[0]
dir_dofs = []
for n in dir_nodes:
    dir_dofs.extend([3 * n + 0, 3 * n + 1, 3 * n + 2])
dir_vals = np.zeros(len(dir_dofs))

K_np = np.asarray(K)
F_np = np.asarray(F)
for d, v in zip(dir_dofs, dir_vals):
    K_np[d, :] = 0.0
    K_np[:, d] = 0.0
    K_np[d, d] = 1.0
    F_np[d] = v

u = spsolve_numpy(K_np, F_np)
print("u first 6 dof:", u[:6])
```

## 4. scikit-fem との比較（変位）

`src/tests/test_solve.py` では以下を自動で行い、最大誤差を出力します:

1. fluxfem で剛性・荷重を組み立て、Dirichlet を適用して解く。
2. scikit-fem で同じメッシュ・要素・荷重を組み立て、節点/DOF 順序を合わせて同じ Dirichlet を適用し、解く。
3. 両者の解ベクトルを比較（最大差 < 1e-5）し、先頭数個の値と最大差を `print`。

試すには:

```bash
PYTHONPATH=src pytest -s src/tests/test_solve.py
```

## 5. 接触デモ（3つのバリエーション）

同じ「ブロック圧縮 + 剛体床接触」を3通りの解法で実装しています。
目的は active set の挙動や収束特性の違いを比較することです。

- `tutorials/linearelastic_tensile_bar_simplified_contact.py`
  - 反復ごとに active set を更新しながら Newton を進める単純版。
  - line search あり／ヒステリシスあり。
- `tutorials/linearelastic_tensile_bar_simplified_contact_active_set.py`
  - 外側で active set を固定し、内側 Newton を数回回す明示的 active set 版。
  - active set の安定性と収束の分離が観察しやすい。
- `tutorials/linearelastic_tensile_bar_simplified_contact_al.py`
  - Augmented Lagrangian（AL）で接触拘束を強化する版。
  - ラグランジュ乗数の更新により、ペナルティ単独より硬い挙動が得やすい。

出力例:

```
elasticity solve max |u_flux - u_sf|: 1e-7
u_flux first 6 dof: [...]
u_sf   first 6 dof: [...]
```

## 6. Mixed WeakForm の標準スタイル

mixed 系は、読みやすさと拡張性を両立するために次の使い分けを基準にします。

- 単一 space:
  `ctx.test` / `ctx.trial` をそのまま使う
- mixed で名前付き field を引きたい:
  `ctx.bindings["u"]`
- mixed で離散空間を明示したい:
  `ctx.spaces["V"]`

典型形は次のとおりです。

```python
import fluxfem as ff

mixed = ff.MixedSpaces(
    {
        "disp": ff.NamedSpace("V", V),
        "press": ff.NamedSpace("Q", Q),
    }
).to_fe_space()

def momentum(v, u, p):
    pressure = ff.unknown_ref("p_like", space="Q")
    return (...) * ff.dOmega()

residuals = ff.make_mixed_residuals(
    momentum=ff.bind_mixed_residual("disp", momentum, space="V"),
)
```

ポイントは次の 3 つです。

- `ctx.bindings[...]` は「名前に束縛された mixed field」
- `ctx.spaces[...]` は「space key ごとの test/trial/unknown bundle」
- 単純ケースでは短い sugar を残し、space の明示は本当に必要な箇所だけにする

新 API の具体例は [tutorials/coupled_reaction_diffusion_new_api.py](/home/kohei/project/physics/fem/tutorials/coupled_reaction_diffusion_new_api.py) と [tutorials/ch3d_fluxfem_wf_new_api.py](/home/kohei/project/physics/fem/tutorials/ch3d_fluxfem_wf_new_api.py) を参照してください。

`MixedFESpace` を直接作る代わりに、命名レイヤだけを先に切ることもできます。

```python
mixed = ff.MixedSpaces(
    {
        "disp": ff.NamedSpace("V", V),
        "press": ff.NamedSpace("Q", Q),
    }
).to_fe_space()
```

Contact も同じ発想で、public spec を先に置けます。

```python
pair = ff.ContactSpaces(master=master_side, slave=slave_side).to_contact_surface_space()
group = ff.ContactGroupSpaces(master=master_side, slaves=[slave_side_1, slave_side_2]).to_contact_surface_space()
onesided = ff.OneSidedContactSpaces(side=slave_side).to_contact_surface_space()
```

## 7. パフォーマンス計測

`test_assembly.py` では `SectionTimer` を使い、JAX 側は JIT のウォームアップと実行時間を分けて計測しています。`pytest -s` で以下のように出力されます:

```
diffusion timings: { 'fluxfem_diffusion_warmup': ..., 'fluxfem_diffusion_assemble': ..., 'skfem_diffusion_assemble': ... }
elasticity timings: { 'fluxfem_elasticity_warmup': ..., 'fluxfem_elasticity_assemble': ..., 'skfem_elasticity_assemble': ... }
```

## 8. ヘルパーまとめ

- メッシュ: `StructuredHexBox`, `tag_axis_minmax_facets`
- 空間: `make_hex_space(mesh, dim, intorder)`
- 要素ベクトル化: `ElementVector(dim)` を `make_space` で利用（`make_hex_space` がラップ）
- 行列組み立て: `space.assemble_bilinear_form`, `linear_elasticity_form`, `diffusion_form`
- 荷重組み立て: `space.assemble_linear_form`, `constant_body_force_form`, `constant_body_force_vector_form`
- スパース組み立て: `assemble_*` に `sparse=True` で (rows, cols, data, n) 形式（線形フォームは rows, data, n）を返却、`coo_to_csr` または `spsolve_numpy` で CSR にして解ける
- 解法: `spsolve_numpy`（SciPy CSR で解く）、ディリクレ条件: `enforce_dirichlet_dense/sparse` または 縮約 `condense_dirichlet` + `expand_dirichlet_solution`
- JAX 反復解法: `cg_solve`（`FluxSparseMatrix` か COO タプルを matvec として使用）
- 境界荷重ヘルパー: Neumann `add_neumann_load`、Robin `add_robin`、面積計算 `facet_area`
- 大変形（St. Venant–Kirchhoff）: `stvk_residual_form`, `deformation_gradient`, `pk2_st_venant_kirchhoff`

これらを組み合わせることで、JAX で高速な組み立てと、SciPy での解法、scikit-fem との突き合わせをシンプルに回せます。

## 9. 大変形ハイパーエラスティック（カンチレバー例）

St. Venant–Kirchhoff の PK2 を使った大変形解析の例です。`stvk_residual_form` はトータルラグランジュ形式の残差を返し、自動微分でヤコビアンを組み立てて Newton で解きます。

100 × 10 × 10 の棒（X 方向長手）を固定端カンチレバーとして、先端に体積力を与えるイメージ（必要に応じて Neumann 面荷重に置き換えてください）。

```python
import jax.numpy as jnp
from fluxfem import (
    StructuredHexBox,
    make_hex_space,
    assemble_residual,
    assemble_jacobian,
    stvk_residual_form,
    newton_solve,
)

# Mesh: 100 x 10 x 10 with 1×1×1 spacing → 11×11×101 nodes
mesh = StructuredHexBox(nx=100, ny=10, nz=10, lx=100.0, ly=10.0, lz=10.0).build()
space = make_hex_space(mesh, dim=3, intorder=2)  # 3 dof/node

# Material parameters (Lamé)
E, nu = 10.0, 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
params = {"mu": mu, "lam": lam}

# Initial guess
u0 = jnp.zeros(space.n_dofs, dtype=jnp.float32)

# Body force example (e.g., gravity in -Y); replace with Neumann if desired
f_body = jnp.array([0.0, -1e-3, 0.0], dtype=jnp.float32)

def body_force_form(ctx, _params):
    # use test.N ordering [u0,v0,w0, u1,v1,w1, ...]
    load = ctx.test.N[..., None] * f_body[None, None, :]  # (n_q, n_ldofs, 3)
    return load.reshape(load.shape[0], -1)

# Assemble external load vector once (total Lagrangian)
# space 経由で外力ベクトルを組み立て
F_ext = space.assemble_linear_form(body_force_form, params=None, sparse=False)

# Residual = internal - external; wrap stvk_residual_form
def total_residual(ctx, u_elem, params):
    return stvk_residual_form(ctx, u_elem, params) - body_force_form(ctx, None)

u_sol, info = newton_solve(space, total_residual, u0, params, tol=1e-6, maxiter=15)
print("Newton info:", info)
```

## 10. 3D 拡散 + メッシュ改善 proxy（厳密解あり）

厳密解付きの 3D Poisson を解き、座標微分で
「真の誤差」と「ZZ 回復 proxy」の方向を比較します。

```bash
python tutorials/diffusion_3d_mesh_proxy.py --nx 6 --ny 6 --nz 6 --perturb 0.1
```

結果は `result/tutorials/diffusion_3d_mesh_proxy` に保存されます。
`gradients_slice.png` に 1) 真の勾配 2) proxy 勾配 を並べた図が出ます。

メモ:
- 先端の境界条件は `add_neumann_load` を使えば面荷重として与えられます。
- 大変形ではステップ制御やダンピングが必要になることが多いので、必要に応じて外側でステップロードやラインサーチを組んでください。
