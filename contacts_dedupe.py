#!/usr/bin/env python3
"""Deduplicate contacts in a vCard (.vcf) export.

Typical phone workflow:
  1. Export contacts as .vcf from the phone / Google Contacts / iCloud.
  2. Run this script (dry-run first).
  3. Import the cleaned .vcf back (or replace the Google Contacts list).

Merge key priority: normalized phone > email > full name (FN).
When duplicates merge, phone/email/org/note fields are unioned; FN keeps the
longest non-empty value.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PHONE_RE = re.compile(r"[^\d+]")
EMAIL_RE = re.compile(r"^EMAIL", re.I)
TEL_RE = re.compile(r"^TEL", re.I)
FN_RE = re.compile(r"^FN", re.I)
N_RE = re.compile(r"^N(?![A-Z])", re.I)
ORG_RE = re.compile(r"^ORG", re.I)
NOTE_RE = re.compile(r"^NOTE", re.I)
BEGIN = "BEGIN:VCARD"
END = "END:VCARD"


def unfold(text: str) -> str:
    """Undo RFC 6350 line folding (CRLF + space/tab continuation)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n[ \t]", "", text)


def normalize_phone(raw: str) -> str:
    digits = PHONE_RE.sub("", raw)
    if digits.startswith("+86") and len(digits) >= 13:
        digits = digits[3:]
    if digits.startswith("86") and len(digits) == 13:
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 12 and digits[1:].isdigit():
        # keep as-is for landlines; still comparable after strip of spaces
        pass
    return digits


def normalize_email(raw: str) -> str:
    return raw.strip().lower()


def normalize_name(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower())


@dataclass
class Contact:
    lines: list[str] = field(default_factory=list)
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    fn: str = ""
    n: str = ""
    orgs: set[str] = field(default_factory=set)
    notes: set[str] = field(default_factory=set)
    other: list[str] = field(default_factory=list)

    def merge_key(self) -> tuple[str, str]:
        if self.phones:
            return ("phone", sorted(self.phones)[0])
        if self.emails:
            return ("email", sorted(self.emails)[0])
        if self.fn:
            return ("fn", normalize_name(self.fn))
        if self.n:
            return ("n", normalize_name(self.n))
        return ("raw", str(hash(tuple(self.lines))))

    def all_keys(self) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for p in self.phones:
            keys.add(("phone", p))
        for e in self.emails:
            keys.add(("email", e))
        if self.fn:
            keys.add(("fn", normalize_name(self.fn)))
        return keys or {self.merge_key()}


def parse_property(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "", line
    name, value = line.split(":", 1)
    prop = name.split(";")[0].upper()
    return prop, value


def parse_vcard_block(block_lines: list[str]) -> Contact:
    c = Contact(lines=list(block_lines))
    for line in block_lines:
        upper = line.upper()
        if upper.startswith(BEGIN) or upper.startswith(END) or upper.startswith("VERSION"):
            continue
        prop, value = parse_property(line)
        if not prop:
            c.other.append(line)
            continue
        if TEL_RE.match(prop):
            norm = normalize_phone(value)
            if norm:
                c.phones.add(norm)
            c.other.append(line)
        elif EMAIL_RE.match(prop):
            norm = normalize_email(value)
            if norm:
                c.emails.add(norm)
            c.other.append(line)
        elif FN_RE.match(prop):
            c.fn = value
            c.other.append(line)
        elif N_RE.match(prop):
            c.n = value
            c.other.append(line)
        elif ORG_RE.match(prop):
            if value.strip():
                c.orgs.add(value.strip())
            c.other.append(line)
        elif NOTE_RE.match(prop):
            if value.strip():
                c.notes.add(value.strip())
            c.other.append(line)
        else:
            c.other.append(line)
    return c


def split_vcards(text: str) -> list[Contact]:
    text = unfold(text)
    contacts: list[Contact] = []
    current: list[str] = []
    in_card = False
    for line in text.split("\n"):
        if not line.strip():
            continue
        upper = line.upper()
        if upper.startswith(BEGIN):
            in_card = True
            current = [line]
            continue
        if not in_card:
            continue
        current.append(line)
        if upper.startswith(END):
            contacts.append(parse_vcard_block(current))
            current = []
            in_card = False
    return contacts


def merge_contacts(group: list[Contact]) -> Contact:
    if len(group) == 1:
        return group[0]

    primary = max(group, key=lambda c: (len(c.fn), len(c.other), len(c.phones)))
    merged = Contact()
    merged.fn = max((c.fn for c in group), key=len)
    merged.n = max((c.n for c in group), key=len) or primary.n
    for c in group:
        merged.phones |= c.phones
        merged.emails |= c.emails
        merged.orgs |= c.orgs
        merged.notes |= c.notes

    # Rebuild a clean vCard rather than concatenating raw duplicates.
    lines = ["BEGIN:VCARD", "VERSION:3.0"]
    if merged.fn:
        lines.append(f"FN:{merged.fn}")
    if merged.n:
        lines.append(f"N:{merged.n}")
    for org in sorted(merged.orgs):
        lines.append(f"ORG:{org}")
    for phone in sorted(merged.phones):
        lines.append(f"TEL;TYPE=CELL:{phone}")
    for email in sorted(merged.emails):
        lines.append(f"EMAIL;TYPE=INTERNET:{email}")
    for note in sorted(merged.notes):
        lines.append(f"NOTE:{note}")
    lines.append("END:VCARD")
    merged.lines = lines
    return merged


def union_find_dedupe(contacts: list[Contact]) -> tuple[list[Contact], list[list[Contact]]]:
    """Union contacts that share any phone/email; fall back to FN-only groups."""
    parent = list(range(len(contacts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    index: dict[tuple[str, str], int] = {}
    for i, c in enumerate(contacts):
        # Prefer phone/email linking; FN alone only links exact same FN when no ids
        strong = {k for k in c.all_keys() if k[0] in ("phone", "email")}
        keys = strong if strong else c.all_keys()
        for key in keys:
            if key in index:
                union(i, index[key])
            else:
                index[key] = i

    groups: dict[int, list[Contact]] = OrderedDict()
    for i, c in enumerate(contacts):
        root = find(i)
        groups.setdefault(root, []).append(c)

    dup_groups = [g for g in groups.values() if len(g) > 1]
    merged = [merge_contacts(g) for g in groups.values()]
    return merged, dup_groups


def contact_label(c: Contact) -> str:
    name = c.fn or c.n or "(无名)"
    phones = ",".join(sorted(c.phones)) or "-"
    return f"{name} | tel={phones}"


def write_vcf(contacts: Iterable[Contact], path: Path) -> None:
    body = "\n".join("\n".join(c.lines) for c in contacts) + "\n"
    path.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove duplicate contacts from a .vcf export."
    )
    parser.add_argument("input", type=Path, help="Input .vcf path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .vcf path (default: <input>_deduped.vcf)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print duplicate groups; do not write output",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"文件不存在: {args.input}", file=sys.stderr)
        return 1

    text = args.input.read_text(encoding="utf-8", errors="replace")
    contacts = split_vcards(text)
    if not contacts:
        print("未解析到任何 vCard，请确认文件是 .vcf 通讯录导出。", file=sys.stderr)
        return 1

    merged, dup_groups = union_find_dedupe(contacts)
    removed = len(contacts) - len(merged)

    print(f"原始联系人: {len(contacts)}")
    print(f"去重后:     {len(merged)}")
    print(f"合并删除:   {removed}")
    print(f"重复组数:   {len(dup_groups)}")

    if dup_groups:
        print("\n重复组合并预览:")
        for i, group in enumerate(dup_groups, 1):
            print(f"  [{i}] {len(group)} 条 →")
            for c in group:
                print(f"      - {contact_label(c)}")

    if args.dry_run:
        print("\n(dry-run，未写入文件)")
        return 0

    out = args.output or args.input.with_name(args.input.stem + "_deduped.vcf")
    write_vcf(merged, out)
    print(f"\n已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
