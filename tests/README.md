# Self-checks

These scripts need no GPU and no data. They are what `make test` runs inside the
container.

| Script | Needs | What it checks |
|---|---|---|
| `test_math.py` | numpy | cosine-schedule endpoints and monotonicity, the variance-preserving kernel by Monte-Carlo, the BFN accuracy identity `sum_i alpha_i = sigma1^-2 - 1` for S in {10,50,1000}, normalisation of the discrete reverse posterior (and its identity for a zero-length step), the DDPM posterior-coefficient identity against the marginal `q(x_s|x_0)` |
| `test_tokenizer.py` | nothing | SMILES tokenisation round-trip on 11 strings, including `[C@@H]`, `%10`, `Cl`, aromatic atoms and stereo bonds |
| `ast_fingerprint.py` | nothing | structural snapshot/diff of a Python tree: hashes the parsed syntax tree with docstrings removed and string literals blanked, so comments and prose may change while any change to an expression, literal or default value is caught |

From the project root:

```bash
python tests/test_math.py
python tests/test_tokenizer.py

# structural snapshot, and later a comparison against it
python tests/ast_fingerprint.py molgen/geobfn /tmp/geobfn_before.json
python tests/ast_fingerprint.py molgen/geobfn /tmp/geobfn_after.json \
       --compare /tmp/geobfn_before.json
```

`ast_fingerprint.py` was used during the project to show that translating the
comments of the GeoBFN branch did not touch a single executable line
(`identical=11 changed=0`). The cross-reference audit of the report itself lives
outside this package, in `../tools/ref_audit.py`, because it checks the LaTeX
sources rather than the code.
