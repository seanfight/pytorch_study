# 通讯录去重

没法直接连你的手机。用下面任一方式即可删掉重复联系人。

## 方式 A（推荐）：Google 通讯录网页合并

安卓机若已同步 Google 账号：

1. 电脑打开 [contacts.google.com](https://contacts.google.com)
2. 左侧点 **合并与修复**（Merge & fix）
3. 按提示合并重复项，或点 **全部合并**
4. 手机端打开「通讯录」下拉刷新，同步后重复会消失

华为/小米/OPPO 等若主要用厂商账号，到各自云服务（华为云空间、小米云等）的网页版找「合并重复联系人」。

## 方式 B：导出 VCF → 本脚本去重 → 再导入

```bash
# 先预览会合并哪些（不写文件）
python3 contacts_dedupe.py 你的通讯录.vcf --dry-run

# 生成去重后的文件
python3 contacts_dedupe.py 你的通讯录.vcf -o contacts_clean.vcf
```

规则：同一**手机号**（会去掉 `+86`、空格、横线）或同一**邮箱**视为同一人，合并姓名/号码/邮箱/公司/备注。

### 从手机导出 / 导入

| 系统 | 导出 | 导入去重后的文件 |
|------|------|------------------|
| Android Google | 通讯录 → 设置 → 导出 | Google Contacts 网页 → 导入；或清空后再导入（慎用） |
| 华为 | 通讯录 → 设置 → 导入/导出 | 同路径导入 `contacts_clean.vcf` |
| iPhone | iCloud.com → 通讯录 → 齿轮 → 导出 vCard | 先在 iCloud 删重复或清空后再导入 |

**注意：** 导入前建议先备份原 `.vcf`。若手机里已有整份通讯录，直接再导入可能又产生重复——更稳妥是：先备份 → 清空手机通讯录（或只用 Google 网页作为主库）→ 导入 `contacts_clean.vcf` → 再同步回手机。

## 测试

```bash
python3 -m unittest test_contacts_dedupe.py -v
```
