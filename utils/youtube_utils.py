import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Extract the video id from video url
def extract_video_id(url):
    # Regex to match standard YouTube URL patterns
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en","hi"])
        transcript = " ".join([chunk["text"] for chunk in transcript_list])
        return transcript
    except TranscriptsDisabled:
        raise Exception("Captions are disabled for this video.")