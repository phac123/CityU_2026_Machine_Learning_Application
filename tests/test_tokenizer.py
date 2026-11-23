"""Standalone sanity check for the SMILES tokenizer (no torch required)."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tok", os.path.join(HERE, "..", "molgen", "llm", "tokenizer.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TESTS = ["CCO", "c1ccccc1", "CC(=O)O", "[nH]1cccc1", "C/C=C/C", "ClCBr",
         "C%10CC%10", "[C@@H](N)C(=O)O", "O=C=O", "N#N", "c1ccc2ccccc2c1"]

tokenizer = mod.SmilesTokenizer.build(TESTS)
ok = True
for smi in TESTS:
    toks = mod.split_smiles(smi)
    ids = tokenizer.encode(smi)
    back = tokenizer.decode(ids)
    status = "OK " if back == smi else "FAIL"
    ok = ok and back == smi
    print(f"{status} {smi:20s} {str(toks):46s} -> {back}")
print("vocab size:", len(tokenizer), "| head:", tokenizer.itos[:10])
print("ALL OK" if ok else "SOME FAILED")
