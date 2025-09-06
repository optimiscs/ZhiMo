# YouTube 视频搜索与中文字幕提取脚本 (README-zh.md)

这是一个 Python 脚本，用于根据提供的标题列表在 YouTube 上搜索视频。它会筛选最近一年内上传的视频，优先选择观看次数最高的，并尝试提取这些视频的详细元数据和**中文 VTT 字幕**。脚本使用 `yt-dlp` 命令行工具执行实际的提取操作，并将结果合并/追加到指定的 JSON 文件中。

## 功能

*   接收一个或多个搜索标题作为输入。
*   筛选最近 **一年** 内上传的视频。
*   为每个标题搜索多个候选视频，并根据**观看次数**进行降序排序。
*   按观看次数从高到低尝试处理候选视频，直到为每个标题找到指定数量 (`--max-results`) 的、**成功提取到中文字幕**的视频为止。
*   **优先提取中文 VTT 字幕** (`zh-Hans`, `zh-CN`, `zh`)。如果某个视频没有找到可用的中文字幕，则会跳过该视频，继续尝试下一个候选视频。
*   提取成功视频的详细元数据，包括：
    *   视频 ID、标题、URL
    *   频道信息（名称、ID、URL）
    *   上传日期、时长、观看次数、点赞次数
    *   视频描述
*   使用 `yt-dlp` **命令行工具** (`subprocess.run`) 来获取信息和下载字幕。
*   支持使用本地浏览器的 Cookies 文件进行身份验证（可能需要访问某些视频）。
*   将提取到的元数据和中文字幕内容整合。
*   将结果**合并/追加**到指定的 JSON 输出文件（默认为 `analyze.json`）。如果文件已存在且包含有效的 JSON 对象，新结果会合并进去；否则会创建新文件或覆盖无效文件。

## 依赖

*   **Python 3**
*   **yt-dlp**: 一个用于从 YouTube 和其他网站下载视频/提取信息的命令行工具。**脚本不自动安装，需要手动安装。**
*   **python-dateutil**: 用于处理日期计算（例如计算一年前的日期）。**脚本不自动安装，需要手动安装。**

## 安装

在运行脚本之前，请确保已安装所需依赖：

```bash
pip install yt-dlp python-dateutil
```

同时，请确保你的系统上安装了 `yt-dlp` 可执行程序，并且可以在命令行中调用。

## 使用方法

通过命令行运行脚本。

**基本语法:**

```bash
python youtube.py --titles "搜索标题1" "搜索标题2" ... [其他选项]
```

**参数说明:**

*   `--titles` / `-t` (必需): 一个或多个要在 YouTube 上搜索的标题。如果标题包含空格，请使用引号括起来。
*   `--max-results` / `-m` (可选): 为**每个**标题尝试获取的、带有中文字幕的、观看次数最高的视频数量。默认为 `1`。脚本会按观看次数处理更多候选视频，直到找到足够数量的成功结果或没有更多候选视频为止。
*   `--output` / `-o` (可选): 指定保存或合并结果的 JSON 文件的路径。默认为 `analyze.json`。
*   `--cookies` (可选): 指定用于身份验证的浏览器 Cookies 文件路径。格式通常为 `"浏览器名称:配置文件路径"`。脚本中的默认值是 Firefox 的一个示例路径，你需要根据你的系统和浏览器进行修改。
    *   示例 (Windows Firefox 默认路径): `"firefox:C:\\Users\\你的用户名\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles\\你的配置文件夹.default-release"`
    *   示例 (macOS Chrome 默认路径): `"chrome:~/.config/google-chrome/Default/Cookies"` (路径可能变化)
    *   **注意:** 直接使用 cookies 文件可能不如 `--cookies-from-browser` 选项稳定，具体取决于 `yt-dlp` 的版本和浏览器实现。请参考 `yt-dlp` 文档获取最新信息。

**示例:**

1.  搜索“热点新闻”和“科技新闻”，每个标题获取观看次数最高的 1 个带中文字幕的视频，并使用 Firefox 的 cookies 合并结果到 `analyze.json`:
    ```bash
    python youtube.py --titles "热点新闻" "科技新闻" --cookies "firefox:C:\\Users\\你的用户名\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles\\你的配置文件夹.default-release"
    ```
    *(请将 cookies 路径替换为你自己的实际路径)*

2.  搜索“AI 最新进展”，获取最多 2 个结果，并将结果保存到 `ai_youtube_data.json` (不使用 cookies):
    ```bash
    python youtube.py --titles "AI 最新进展" --max-results 2 --output ai_youtube_data.json
    ```

## 输出

脚本执行后，会更新或创建指定的 JSON 文件（默认为 `analyze.json`）。该文件包含一个 JSON 对象 (字典)，其中：

*   **键 (Key)**: 是你输入的搜索标题。
*   **值 (Value)**: 是一个包含该标题下成功找到的视频数据的列表。列表中的每个元素是一个字典，包含以下结构：

```json
{
  "搜索标题1": [
    {
      "video_info": {
        "title": "视频标题",
        "video_id": "视频ID",
        "url": "视频URL",
        "channel": "频道名称",
        "channel_id": "频道ID",
        "channel_url": "频道URL",
        "upload_date": "YYYYMMDD",
        "duration": 视频时长（秒）,
        "view_count": 观看次数,
        "like_count": 点赞次数,
        "description": "视频描述"
      },
      "vtt_content": "提取到的中文字幕内容 (VTT 格式字符串)"
    },
    // ... 可能有更多该标题下的视频 (如果 max-results > 1)
  ],
  "搜索标题2": [
    // ... 标题2 的视频数据 ...
  ]
  // ... 其他标题
}
```

## 注意事项

*   脚本强依赖 **中文 VTT 字幕**。如果按观看次数排序的候选视频没有提供中文 VTT 字幕，脚本会跳过该视频，继续尝试下一个，直到找到满足 `max-results` 数量的带中文字幕的视频或候选视频处理完毕。
*   脚本通过调用 `yt-dlp` **命令行程序**来工作，而不是纯粹的 Python API 调用。请确保 `yt-dlp` 在系统路径中可执行。
*   输出文件采用**合并/追加**模式。如果输出文件已存在且是有效的 JSON 对象，新搜索到的标题及其视频列表会被添加进去，或者更新已存在标题的视频列表（当前实现是追加）。
*   访问某些受限视频或获取更可靠的信息可能需要有效的**浏览器 Cookies**。
*   如果脚本执行出错，它会打印一些可能的解决方案提示，请根据提示检查配置。
*   视频的观看次数、点赞数等信息依赖于 `yt-dlp` 能否成功提取。