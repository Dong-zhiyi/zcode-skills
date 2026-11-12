# zcode-skills

我的 ZCode 自建技能（skills）集合。

每个技能是一个 ZCode 可自动发现的目录：`SKILL.md` 的 frontmatter 声明技能名与触发条件，可附带 `scripts/`、`assets/` 目录放配套脚本或资源。

## 技能列表

| 技能 | 说明 |
| --- | --- |
| [image-watermark-removal](skills/image-watermark-removal/) | 去除图片中的水印（logo 标、网站名、「即梦AI」类 AI 工具标、平铺/半透明水印、角落标记）。按亮度差定位标记区域，OpenCV NS 修复 + 纹理迁移 |

## 安装

克隆后，把需要的技能目录复制到 ZCode 的技能目录：

- **用户级**（所有项目生效）：`~/.zcode/skills/`（Windows：`C:\Users\<用户名>\.zcode\skills\`）
- **项目级**（仅当前项目生效）：`<项目目录>/.zcode/skills/`

示例（Windows Git Bash）：

```bash
git clone https://github.com/Dong-zhiyi/zcode-skills.git
cp -r zcode-skills/skills/image-watermark-removal ~/.zcode/skills/
```

然后重启 ZCode 客户端（或新开会话），技能会自动出现在可用技能列表中。

## 脚本依赖

部分技能带配套脚本，使用前先装好依赖：

- `image-watermark-removal`：`pip install numpy opencv-python pillow`
