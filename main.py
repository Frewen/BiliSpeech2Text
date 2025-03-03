import os
import re
import sys
import json
import time
import requests
from bs4 import BeautifulSoup
import ffmpeg
from pydub import AudioSegment
import whisper
from tqdm import tqdm
from typing import List, Dict

class BiliDown2Text:
    # 用于提取页面中的__INITIAL_STATE__数据的正则表达式
    INITIAL_STATE_PATTERN = r'window\.__INITIAL_STATE__=([\s|\S]+);\(function'
    # 用于提取视频ID的正则表达式
    AV_PATTERN = r'av(\d+)'
    BV_PATTERN = r'BV([a-zA-Z0-9]+)'

    @staticmethod
    def is_cuda_available():
        return whisper.torch.cuda.is_available()

    def __init__(self, model_size='small'):
        self.model = whisper.load_model(model_size, device="cuda" if self.is_cuda_available() else "cpu")
        self.base_dir = 'bilibili_video'
        self.config = self.load_config()
        self.setup_directories()

    def load_config(self):
        """加载配置文件"""
        config_path = 'config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f'加载配置文件失败: {str(e)}')
        return {'bilibili': {'cookies': {}, 'headers': {}}}

    def setup_directories(self):
        """创建基础目录"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    @staticmethod
    def get_initial_state(html_content: str) -> str:
        """从HTML内容中提取__INITIAL_STATE__数据"""
        match = re.search(BiliDown2Text.INITIAL_STATE_PATTERN, html_content)
        if not match:
            return ''
        return match.group(1)

    def check_collection(self, url: str) -> tuple[bool, dict]:
        """检查视频是否为合集，返回(是否合集, 视频数据)的元组"""
        try:
            headers = self.config['bilibili']['headers']
            cookies = self.config['bilibili']['cookies']
            response = requests.get(url, headers=headers, cookies=cookies)
            response.raise_for_status()

            html_content = response.text
            initial_state = self.get_initial_state(html_content)
            if not initial_state:
                return False, {}

            try:
                data = json.loads(initial_state)
                video_data = data.get('videoData', {})
                ugc_season = video_data.get('ugc_season')
                if ugc_season:
                    return True, data
                return False, data
            except json.JSONDecodeError:
                print('解析页面数据失败')
                return False, {}

        except Exception as e:
            print(f'检查合集状态时发生错误: {str(e)}')
            return False, {}

    def _load_progress(self, collection_id: str) -> set:
        """加载下载进度"""
        progress_file = os.path.join(self.base_dir, f'progress_{collection_id}.json')
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception as e:
                print(f'加载进度文件失败: {str(e)}')
        return set()

    def _save_progress(self, collection_id: str, downloaded_videos: set) -> None:
        """保存下载进度"""
        progress_file = os.path.join(self.base_dir, f'progress_{collection_id}.json')
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(list(downloaded_videos), f)
        except Exception as e:
            print(f'保存进度文件失败: {str(e)}')

    def process_collection(self, data: dict) -> List[Dict]:
        """处理合集数据，返回选中的视频列表"""
        video_data = data.get('videoData', {})
        ugc_season = video_data.get('ugc_season', {})
        sections = ugc_season.get('sections', [])
        collection_id = video_data.get('bvid', '')

        # 加载已下载的视频记录
        downloaded_videos = self._load_progress(collection_id)

        # 显示合集信息
        print('\n' + '='*50)
        print('检测到视频合集')
        print('='*50)
        print(f'合集标题：{ugc_season.get("title", "未知标题")}')
        total_episodes = sum(len(section.get('episodes', [])) for section in sections)
        print(f'合集总集数：{total_episodes}')
        print('-'*50 + '\n')

        # 显示分集信息
        video_index = 1  # 添加序号计数
        for section_index, section in enumerate(sections, 1):
            if len(sections) > 1:
                print(f'第{section_index}部分：{section.get("title", "未命名部分")}')                
                print('-'*30)
            episodes = section.get('episodes', [])
            for episode in episodes:
                title = episode.get('title', '未知标题')
                long_title = episode.get('long_title', '')
                bvid = episode.get('bvid', '')
                if long_title:
                    title = f'{title} - {long_title}'
                print(f'[{video_index}] [{bvid}] {title}')
                video_index += 1
            if len(sections) > 1:
                print()
        print('='*50)

        # 选择下载选项
        print('\n请选择下载选项：')
        print('1. 下载整个合集')
        print('2. 选择单个或多个视频')
        print('3. 按区间下载')
        choice = input('请输入选项编号 (1/2/3): ')

        selected_videos = []
        if choice == '1':
            # 下载整个合集
            for section in sections:
                for episode in section.get('episodes', []):
                    bvid = episode.get('bvid', '')
                    if bvid in downloaded_videos:
                        print(f'跳过已下载的视频：{bvid}')
                        continue
                    
                    title = episode.get('title', '未知标题')
                    long_title = episode.get('long_title', '')
                    if long_title:
                        title = f'{title} - {long_title}'
                    if bvid:
                        selected_videos.append({
                            'title': title,
                            'bvid': bvid,
                            'aid': episode.get('aid')
                        })
        elif choice == '2':
            # 选择特定视频
            print('\n请输入要下载的视频序号或BV号（多个用逗号分隔）：')
            selection = input('> ').strip().replace('，', ',')
            selected_bvids = [x.strip() for x in selection.split(',') if x.strip()]

            # 构建视频映射表
            video_map = {}
            index = 1
            for section in sections:
                for episode in section.get('episodes', []):
                    bvid = episode.get('bvid', '')
                    title = episode.get('title', '未知标题')
                    long_title = episode.get('long_title', '')
                    if long_title:
                        title = f'{title} - {long_title}'
                    if bvid:
                        video_info = {
                            'title': title,
                            'bvid': bvid,
                            'aid': episode.get('aid')
                        }
                        video_map[str(index)] = video_info
                        video_map[bvid] = video_info
                        index += 1

            for selection in selected_bvids:
                if selection in video_map:
                    selected_videos.append(video_map[selection])
        elif choice == '3':
            # 按区间下载
            print('\n请输入下载区间的起始和结束序号（如：1,10）：')
            try:
                start, end = map(int, input('> ').strip().split(','))
                # 构建视频列表
                all_videos = []
                for section in sections:
                    for episode in section.get('episodes', []):
                        bvid = episode.get('bvid', '')
                        title = episode.get('title', '未知标题')
                        long_title = episode.get('long_title', '')
                        if long_title:
                            title = f'{title} - {long_title}'
                        if bvid:
                            all_videos.append({
                                'title': title,
                                'bvid': bvid,
                                'aid': episode.get('aid')
                            })
                
                # 验证区间范围
                if 1 <= start <= end <= len(all_videos):
                    for i in range(start-1, end):
                        video = all_videos[i]
                        if video['bvid'] not in downloaded_videos:
                            selected_videos.append(video)
                        else:
                            print(f'跳过已下载的视频：{video["bvid"]}')
                else:
                    print(f'无效的区间范围，请输入1到{len(all_videos)}之间的数字')
            except (ValueError, IndexError):
                print('输入格式错误，请按照"起始序号,结束序号"的格式输入')

        return selected_videos

    def extract_video_info(self, url: str) -> List[Dict]:
        """从URL中提取视频信息，支持合集和单个视频"""
        # 检查是否为合集
        is_collection, data = self.check_collection(url)
        
        if is_collection and data:
            return self.process_collection(data)
        else:
            # 处理单个视频
            bv_match = re.search(self.BV_PATTERN, url)
            av_match = re.search(self.AV_PATTERN, url, re.I)
            
            if bv_match:
                bvid = f'BV{bv_match.group(1)}'
                api_url = f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}'
            elif av_match:
                aid = av_match.group(1)
                api_url = f'https://api.bilibili.com/x/web-interface/view?aid={aid}'
            else:
                print("无效的视频链接格式")
                return []
                
            try:
                headers = self.config['bilibili']['headers']
                cookies = self.config['bilibili']['cookies']
                response = requests.get(api_url, headers=headers, cookies=cookies)
                data = response.json()
                
                if data['code'] == 0:
                    video_data = data['data']
                    return [{
                        'title': video_data.get('title'),
                        'bvid': video_data.get('bvid'),
                        'aid': video_data.get('aid')
                    }]
                else:
                    print(f"获取视频信息失败：{data.get('message')}")
                    return []
            except Exception as e:
                print(f"请求视频信息时出错：{str(e)}")
                return []

    def _get_playlist_info(self, ugc_season):
        videos = []
        for episode in ugc_season.get('sections', [{}])[0].get('episodes', []):
            videos.append({
                'title': episode.get('title'),
                'bvid': episode.get('bvid'),
                'aid': episode.get('aid')
            })
        return videos

    def download_video(self, video_info):
        title = video_info['title']
        video_id = video_info['bvid']
        
        # 创建目录结构
        base_dir = f"bilibili_video/{title}"
        video_dir = f"{base_dir}/video"
        os.makedirs(video_dir, exist_ok=True)
        
        video_path = f"{video_dir}/{video_id}.mp4"
        if os.path.exists(video_path):
            print(f"视频已存在：{video_path}")
            return video_path

        # 获取视频信息
        headers = self.config['bilibili']['headers']
        cookies = self.config['bilibili']['cookies']
        
        api_url = f'https://api.bilibili.com/x/web-interface/view?bvid={video_id}'
        try:
            response = requests.get(api_url, headers=headers, cookies=cookies)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 0 and 'data' in data:
                # 使用you-get下载视频
                video_url = f'https://www.bilibili.com/video/{video_id}'
                sys.argv = ['you-get', '-o', video_dir, '--output-filename', video_id, video_url]
                try:
                    from you_get import common as you_get
                    you_get.main()
                except Exception as e:
                    print(f"下载视频时发生错误：{str(e)}")
                    return None
            else:
                print(f'获取视频信息失败：{data.get("message", "未知错误")}')
                return None
        except Exception as e:
            print(f"获取视频信息时发生错误：{str(e)}")
            return None
        
        # 检查下载后的文件是否存在，可能文件名与预期不同
        if not os.path.exists(video_path):
            # 查找可能的文件
            files = os.listdir(video_dir)
            for file in files:
                if file.endswith('.mp4') and (video_id in file or title in file):
                    # 找到匹配的文件，重命名为预期的文件名
                    src_path = os.path.join(video_dir, file)
                    os.rename(src_path, video_path)
                    break
        
        if os.path.exists(video_path):
            print(f"视频下载完成：{video_path}")
            return video_path
        else:
            print("视频下载失败")
            return None

    def extract_audio(self, video_path, title, video_id):
        audio_dir = f"bilibili_video/{title}/conv"
        os.makedirs(audio_dir, exist_ok=True)
        
        audio_path = f"{audio_dir}/{video_id}.mp3"
        if os.path.exists(audio_path):
            print(f"音频已存在：{audio_path}")
            return audio_path
        
        # 使用ffmpeg提取音频，不输出日志
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, loglevel='quiet')
        ffmpeg.run(stream)
        
        print(f"音频提取完成：{audio_path}")
        return audio_path

    def split_audio(self, audio_path, title, video_id):
        slice_dir = f"bilibili_video/{title}/slice/{video_id}"
        os.makedirs(slice_dir, exist_ok=True)
        
        # 加载音频文件
        audio = AudioSegment.from_mp3(audio_path)
        chunk_length = 45000  # 45秒
        chunks = []
        
        # 分割音频
        for i, chunk_start in enumerate(range(0, len(audio), chunk_length)):
            chunk = audio[chunk_start:chunk_start + chunk_length]
            chunk_path = f"{slice_dir}/{i}.mp3"
            chunk.export(chunk_path, format="mp3")
            chunks.append(chunk_path)
        
        return chunks

    def transcribe_audio(self, audio_chunks, title, video_id):
        output_dir = f"bilibili_video/{title}/outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f"{output_dir}/{video_id}.txt"
        texts = []
        initial_prompt = f"以下是普通话的句子。这是一个关于{title}的视频。"
        
        for chunk_path in tqdm(audio_chunks, desc="转换音频为文本"):
            result = self.model.transcribe(chunk_path, initial_prompt=initial_prompt, fp16=False)
            text = result["text"].strip()
            texts.append(text)
            print(f"转换结果：{text}")
        
        # 保存文本文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
        
        print(f"文本转换完成：{output_path}")
        return output_path

    def process_video(self, url):
        # 获取视频信息
        videos = self.extract_video_info(url)
        if not videos:
            print("无法获取视频信息")
            return
        
        # 获取合集ID（如果是合集的话）
        is_collection, data = self.check_collection(url)
        collection_id = data.get('videoData', {}).get('bvid', '') if is_collection else ''
        downloaded_videos = self._load_progress(collection_id) if collection_id else set()
        
        for video in videos:
            print(f"\n处理视频：{video['title']}")
            
            # 检查文本文件是否已存在
            output_dir = f"bilibili_video/{video['title']}/outputs"
            output_path = f"{output_dir}/{video['bvid']}.txt"
            if os.path.exists(output_path):
                print(f"文本文件已存在：{output_path}，跳过处理")
                continue
            
            # 下载视频
            video_path = self.download_video(video)
            if not video_path:
                print("视频下载失败")
                continue
            
            # 提取音频
            audio_path = self.extract_audio(video_path, video['title'], video['bvid'])
            
            # 分割音频
            audio_chunks = self.split_audio(audio_path, video['title'], video['bvid'])
            
            # 转换为文本
            self.transcribe_audio(audio_chunks, video['title'], video['bvid'])
            
            # 更新进度记录
            if collection_id:
                downloaded_videos.add(video['bvid'])
                self._save_progress(collection_id, downloaded_videos)

def main():
    if len(sys.argv) != 2:
        print("使用方法: python main.py <视频链接>")
        return
    
    url = sys.argv[1]
    downloader = BiliDown2Text()
    downloader.process_video(url)

if __name__ == "__main__":
    main()