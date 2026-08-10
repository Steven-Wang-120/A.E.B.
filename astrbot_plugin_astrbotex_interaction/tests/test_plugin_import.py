from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


class PluginImportTests(unittest.TestCase):
    def test_plugin_registers_astrbotex_platform(self) -> None:
        work_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(work_root))
        module = importlib.import_module("astrbot_plugin_astrbotex_interaction.main")

        from astrbot.core.platform.register import platform_cls_map

        self.assertIs(
            platform_cls_map["astrbotex"],
            module.AstrBotEXPlatformAdapter,
        )
        self.assertEqual(module.DEFAULT_TEXT_PORT, 8766)
        self.assertEqual(module.DEFAULT_AUDIO_PORT, 8767)
        self.assertEqual(module.DEFAULT_VISION_PORT, 8768)


if __name__ == "__main__":
    unittest.main()
