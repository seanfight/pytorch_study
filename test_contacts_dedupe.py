#!/usr/bin/env python3
"""Smoke tests for contacts_dedupe.py (no third-party deps)."""

from pathlib import Path
import tempfile
import unittest

from contacts_dedupe import (
    normalize_phone,
    split_vcards,
    union_find_dedupe,
    write_vcf,
)


SAMPLE = """BEGIN:VCARD
VERSION:3.0
FN:张三
TEL;TYPE=CELL:+86 138-0013-8000
EMAIL:zhangsan@example.com
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:张三
TEL;TYPE=CELL:13800138000
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:李四
TEL;TYPE=CELL:13900139000
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:李四（工作）
TEL;TYPE=WORK:13900139000
EMAIL:lisi@work.com
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:王五
EMAIL:wangwu@example.com
END:VCARD
BEGIN:VCARD
VERSION:3.0
FN:王五备份
EMAIL:wangwu@example.com
END:VCARD
"""


class DedupeTests(unittest.TestCase):
    def test_normalize_phone_cn(self):
        self.assertEqual(normalize_phone("+86 138-0013-8000"), "13800138000")
        self.assertEqual(normalize_phone("13800138000"), "13800138000")
        self.assertEqual(normalize_phone("8613800138000"), "13800138000")

    def test_merge_by_phone_and_email(self):
        contacts = split_vcards(SAMPLE)
        self.assertEqual(len(contacts), 6)
        merged, groups = union_find_dedupe(contacts)
        self.assertEqual(len(merged), 3)
        self.assertEqual(len(groups), 3)

        by_fn = {c.fn: c for c in merged}
        # 张三 / 李四 keep longest FN after merge
        zhang = next(c for c in merged if "13800138000" in c.phones)
        self.assertIn("zhangsan@example.com", zhang.emails)

        li = next(c for c in merged if "13900139000" in c.phones)
        self.assertTrue(li.fn.startswith("李四"))
        self.assertIn("lisi@work.com", li.emails)

        wang = next(c for c in merged if "wangwu@example.com" in c.emails)
        self.assertTrue(len(wang.fn) > 0)

    def test_roundtrip_write(self):
        contacts = split_vcards(SAMPLE)
        merged, _ = union_find_dedupe(contacts)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.vcf"
            write_vcf(merged, out)
            again = split_vcards(out.read_text(encoding="utf-8"))
            self.assertEqual(len(again), 3)


if __name__ == "__main__":
    unittest.main()
