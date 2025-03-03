# BiliSpeech2Text

## 简介
BiliSpeech2Text 是一个自动化工具，用于将 Bilibili 视频转换为文本。它通过下载视频、提取音频、分割音频，并使用 OpenAI 的 Whisper 模型将语音转换为文本。整个过程是自动化的，用户只需提供 Bilibili 视频链接即可。

## 功能
- 支持通过 av 号或 BV 号下载 Bilibili 视频
- 自动提取视频中的音频内容
- 智能分割音频为小段（45秒）以提高处理效率
- 使用 Whisper 模型（默认 small）将语音转换为文本
- 支持视频合集的批量处理和选择性下载
- 智能复用已下载的视频和音频文件，避免重复下载
- 支持自定义初始提示词，提高转换准确度
- 支持 CUDA 加速（如果可用）

## 使用方法
1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 配置（可选）
   - 创建 `config.json` 文件，添加 Bilibili cookies 和 headers 配置

3. 运行脚本
   ```bash
   python main.py <视频链接>
   ```

## 示例
1. 处理单个视频
   ```bash
   python main.py https://www.bilibili.com/video/BV1BUkyYTE9X
   ```

2. 处理视频合集
   - 运行脚本后会显示合集信息
   - 支持三种下载模式：
     - 下载整个合集
     - 选择单个或多个视频
     - 按区间下载

3. 输出说明
   - 视频文件保存在 `bilibili_video/{视频标题}/video/{video_id}.mp4`
   - 音频文件保存在 `bilibili_video/{视频标题}/conv/{video_id}.mp3`
   - 分割后的音频片段保存在 `bilibili_video/{视频标题}/slice/{video_id}/{index}.mp3`
   - 最终文本输出在 `bilibili_video/{视频标题}/outputs/{video_id}.txt`