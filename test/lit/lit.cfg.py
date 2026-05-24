# Suite-wide LIT settings (paths live in lit.site.cfg.py, generated under build/).
import lit.formats

config.name = "ai-compiler"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
